"""Instance-segmentation metrics for the SegVGGT object-query branch.

``src/evaluation/metrics.py`` covers rendering (PSNR/SSIM/LPIPS), pose and depth;
none of it says anything about *instances*.  This module fills that gap so a training
run can answer the only question that matters mid-flight: **is the set-prediction
objective actually being met, or is the loss just wobbling?**

Two families of numbers, deliberately separated:

*Diagnostics* -- cheap, dense, meant to be logged every validation step:
  ``matched_iou_mean``      IoU averaged over an IoU-optimal 1-1 query<->GT assignment.
                            This is the quantity the mask loss is trying to maximise;
                            it should climb from ~0 towards >0.5.  It is the single
                            most informative scalar here.
  ``best_iou_per_gt_mean``  per GT, the best IoU over *all* queries (no assignment).
                            Compare with ``matched_iou_mean``: a large gap means the
                            masks exist but the assignment/classification is off; both
                            near zero means nothing was learned at all.
  ``n_gt`` / ``n_fired``    GT instance count vs. queries with objectness above
                            ``score_threshold``.  ``fired_over_gt`` is their ratio --
                            it catches the two classic collapses (everything predicted
                            no-object, or every query firing).
  ``mask_area_mean``        mean fraction of supervised pixels covered by a fired
                            query's mask.  Watches for mask over-growth.

*Benchmark* -- the ScanNet-style numbers you would put in a table:
  ``ap25`` / ``ap50`` / ``ap``  (the last averaged over IoU 0.50:0.05:0.95)

Conventions, chosen to match training exactly so the numbers are comparable:
  - IoU is computed on the **flattened S*h*w volume**, i.e. the paper's "a multi-view
    mask is a point-cloud mask" identity (see :mod:`src.loss.segvggt_matcher`).
  - GT is area-resampled to the prediction's (h, w) and thresholded at 0.5, the same
    ``LossSegVGGT._downsample_masks`` recipe.
  - ``instance_valid_mask`` pixels are dropped from both intersection and union, the
    same way the loss's ``keep`` drops them.
  - objectness score is ``1 - P(no-object)``, matching both the inference decoder and
    the wrapper's ``val/queries_fired``.

No learnable parameters, no repo dependencies beyond torch (+ scipy for the optimal
assignment, with the same graceful fallback as the matcher).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - metrics stay importable without scipy
    _HAS_SCIPY = False


# IoU thresholds for the averaged AP, ScanNet / COCO convention.
AP_THRESHOLDS: tuple[float, ...] = tuple(0.5 + 0.05 * i for i in range(10))

_EMPTY_METRICS: dict[str, float] = {
    "matched_iou_mean": 0.0,
    "best_iou_per_gt_mean": 0.0,
    "n_gt": 0.0,
    "n_fired": 0.0,
    "fired_over_gt": 0.0,
    "mask_area_mean": 0.0,
    "ap25": 0.0,
    "ap50": 0.0,
    "ap": 0.0,
}


def _downsample_binary(binary: Tensor, hw: tuple[int, int]) -> Tensor:
    """[K, S, H, W] binary -> [K, S, h, w] binary (area resample + 0.5 threshold).

    Mirrors ``LossSegVGGT._downsample_masks`` so metric and loss agree on what a GT
    pixel is.
    """
    k, s, height, width = binary.shape
    h, w = hw
    if (height, width) == (h, w):
        return binary.float()
    x = binary.reshape(k * s, 1, height, width).float()
    x = F.interpolate(x, size=(h, w), mode="area")
    return (x.reshape(k, s, h, w) > 0.5).float()


def _pairwise_iou(pred: Tensor, gt: Tensor, keep: Tensor | None) -> Tensor:
    """IoU between every prediction and every GT.

    pred : [Q, N] binary float ; gt : [K, N] binary float ; keep : [N] float or None.
    Returns [Q, K].
    """
    if keep is not None:
        pred = pred * keep
        gt = gt * keep
    inter = pred @ gt.transpose(0, 1)                       # [Q, K]
    union = pred.sum(-1)[:, None] + gt.sum(-1)[None, :] - inter
    return inter / union.clamp_min(1e-6)


def _average_precision(iou: Tensor, scores: Tensor, n_gt: int, threshold: float) -> float:
    """Greedy score-ordered matching -> all-point-interpolated AP at one IoU threshold.

    ``iou`` [Q, K], ``scores`` [Q].  Every prediction participates (no score cut-off);
    thresholding predictions here would silently inflate precision.
    """
    if n_gt == 0 or iou.numel() == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    taken = torch.zeros(n_gt, dtype=torch.bool)
    tp = torch.zeros(order.numel())
    fp = torch.zeros(order.numel())

    for rank, q in enumerate(order.tolist()):
        candidates = iou[q].clone()
        candidates[taken] = -1.0
        best = int(torch.argmax(candidates))
        if candidates[best] >= threshold:
            tp[rank] = 1.0
            taken[best] = True
        else:
            fp[rank] = 1.0

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp).clamp_min(1e-6)

    # all-point interpolation: make precision monotonically decreasing, then integrate
    mrec = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    mpre = torch.cat([torch.zeros(1), precision, torch.zeros(1)])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def _optimal_assignment(iou: Tensor) -> tuple[Tensor, Tensor]:
    """1-1 assignment maximising total IoU. Returns (query_idx, gt_idx)."""
    cost = (-iou).cpu().numpy()
    if _HAS_SCIPY:
        row, col = linear_sum_assignment(cost)
        return torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)
    # greedy fallback (same spirit as segvggt_matcher._greedy_assignment)
    q, k = iou.shape
    flat = torch.argsort(iou.reshape(-1), descending=True)
    used_q: set[int] = set()
    used_k: set[int] = set()
    rows: list[int] = []
    cols: list[int] = []
    for idx in flat.tolist():
        r, c = divmod(idx, k)
        if r in used_q or c in used_k:
            continue
        used_q.add(r)
        used_k.add(c)
        rows.append(r)
        cols.append(c)
        if len(rows) == min(q, k):
            break
    return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


@torch.no_grad()
def compute_instance_metrics(
    query_masks: Tensor,
    query_class_logits: Tensor,
    instance_mask: Tensor,
    valid_mask: Tensor | None = None,
    *,
    ignore_id: int = 0,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict[str, float]:
    """Instance metrics for **one** sample (no batch dimension).

    Parameters
    ----------
    query_masks : [Q, S, h, w] raw mask logits (pre-sigmoid).
    query_class_logits : [Q, C+1] class logits; the last channel is *no-object*.
    instance_mask : [S, H, W] int64, multi-view-consistent instance ids.
    valid_mask : [S, H, W] bool (or [S, H, W, 1]); False pixels are ignored entirely.
    ignore_id : instance id that does not define an instance (0 by convention).
    score_threshold : objectness cut-off for the "fired" diagnostics only; AP uses
        every query.
    mask_threshold : sigmoid cut-off turning mask logits into a binary mask.

    Returns
    -------
    dict of plain floats, safe to hand straight to ``self.log``.  A scene with no GT
    instances returns all-zeros rather than NaNs.
    """
    q, s, h, w = query_masks.shape
    device = query_masks.device

    # ---- GT instances ----------------------------------------------------
    ids = torch.unique(instance_mask)
    ids = ids[ids != ignore_id]
    k = int(ids.numel())
    if k == 0:
        return dict(_EMPTY_METRICS)

    masks_full = (instance_mask[None] == ids[:, None, None, None]).float()  # [K,S,H,W]
    gt = _downsample_binary(masks_full, (h, w)).reshape(k, s * h * w)

    keep = None
    if valid_mask is not None:
        if valid_mask.shape[-1] == 1 and valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(-1)
        keep = _downsample_binary(valid_mask.to(torch.bool).float()[None], (h, w)).reshape(-1)

    # ---- predictions -----------------------------------------------------
    probs = query_class_logits.float().softmax(-1)
    scores = 1.0 - probs[:, -1]                                    # [Q] objectness
    pred = (query_masks.reshape(q, s * h * w).float().sigmoid() > mask_threshold).float()

    iou = _pairwise_iou(pred, gt, keep)                            # [Q, K]

    # ---- diagnostics -----------------------------------------------------
    q_idx, g_idx = _optimal_assignment(iou)
    matched_iou = iou[q_idx.to(device), g_idx.to(device)]
    fired = scores > score_threshold
    n_fired = float(fired.sum())

    denom = pred.shape[1] if keep is None else float(keep.sum().clamp_min(1.0))
    if n_fired > 0:
        # matmul instead of `pred[fired] * keep` so no second [F, N] tensor is
        # materialised -- N is S*h*w and this runs inside the validation step.
        area = pred[fired].sum(-1) if keep is None else pred[fired] @ keep
        mask_area_mean = float(area.mean() / denom)
    else:
        mask_area_mean = 0.0

    metrics = {
        "matched_iou_mean": float(matched_iou.mean()) if matched_iou.numel() else 0.0,
        "best_iou_per_gt_mean": float(iou.max(dim=0).values.mean()),
        "n_gt": float(k),
        "n_fired": n_fired,
        "fired_over_gt": n_fired / k,
        "mask_area_mean": mask_area_mean,
    }

    # ---- benchmark AP ----------------------------------------------------
    iou_cpu = iou.detach().cpu()
    scores_cpu = scores.detach().cpu()
    metrics["ap25"] = _average_precision(iou_cpu, scores_cpu, k, 0.25)
    aps = [_average_precision(iou_cpu, scores_cpu, k, t) for t in AP_THRESHOLDS]
    metrics["ap50"] = aps[0]
    metrics["ap"] = sum(aps) / len(aps)
    return metrics


@torch.no_grad()
def compute_instance_metrics_batch(
    query_masks: Tensor,
    query_class_logits: Tensor,
    instance_mask: Tensor,
    valid_mask: Tensor | None = None,
    **kwargs,
) -> dict[str, float]:
    """Batched wrapper: averages :func:`compute_instance_metrics` over B samples.

    Shapes carry a leading batch dim ([B, Q, S, h, w] etc.).  Scenes with no GT
    instance contribute nothing (they are skipped, not averaged in as zeros).
    """
    b = query_masks.shape[0]
    rows: list[dict[str, float]] = []
    for i in range(b):
        m = compute_instance_metrics(
            query_masks[i],
            query_class_logits[i],
            instance_mask[i],
            None if valid_mask is None else valid_mask[i],
            **kwargs,
        )
        if m["n_gt"] > 0:
            rows.append(m)
    if not rows:
        return dict(_EMPTY_METRICS)
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}

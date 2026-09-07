"""Physics-property metrics for the SegVGGT object-query branch.

``instance_metrics.py`` answers "is the set-prediction objective being met".
This module answers the question the map's destination is actually about:
**how close is the student to the teacher, relative to a class lookup table.**

Why not the training loss.  The head is trained with a Gaussian NLL on z-scored
targets; NLL mixes the point estimate with the learned variance and is not
reportable.  What goes in a table is a per-property error in the units the
property is quoted in:

  * ``density`` / ``youngs_modulus``: MAE in **log10 SI** (log10 kg/m³, log10 Pa)
  * ``poisson_ratio``: MAE in **raw** units (it is not log-transformed)

Both come free from z-space: a z-score is ``(x - mean) / std`` with a *fixed*
std per property, so ``|x_pred - x_gt| = std * |z_pred - z_gt|``.  No
de-normalisation round-trip, no clamping asymmetry.

Three rows, all scored on **the same matched instances** so they are comparable:

  ``phys_mae_*``        the student
  ``phys_mae_*_const``  predict the training mean for everything (z = 0) -- the floor
  ``phys_mae_*_clut``   predict the training mean of the instance's class -- the
                        real reference frame (ticket 03).  Needs
                        ``PhysGMTarget.class_mean_lut``; silently absent otherwise.

The teacher is deliberately not a row: the student's labels *are* the teacher,
so its error is 0 by construction.

Matching.  Queries are assigned to GT instances by the same IoU-optimal
assignment ``matched_iou_mean`` uses (imported from ``instance_metrics`` rather
than reimplemented), on the same flattened S*h*w volume.  One deliberate
difference: GT instances the teacher never labelled are dropped *before* the
assignment, since they have no target to score against -- so this is the same
recipe over a subset, not literally the same assignment as
``matched_iou_mean``.  Either way the physics numbers inherit the segmentation
numbers' bias, notably the pull towards large easy objects early in training.
``phys_n_matched`` records how many instances were scored.
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.dataset.physics.parsers import PHYSGM_NORMALIZATION
from src.dataset.physics.types import PhysGMTarget

from .instance_metrics import _downsample_binary, _optimal_assignment, _pairwise_iou

# Reporting unit per property, mirroring PHYSGM_NORMALIZATION's log10 flag.
_UNIT_SUFFIX: tuple[str, ...] = ("log10", "log10", "raw")


def _metric_name(prop: str, i: int) -> str:
    return f"phys_mae_{_UNIT_SUFFIX[i]}_{prop}"


@torch.no_grad()
def compute_physics_metrics(
    query_masks: Tensor,
    query_phys_mu: Tensor,
    instance_mask: Tensor,
    target: PhysGMTarget,
    valid_mask: Tensor | None = None,
    *,
    ignore_id: int = 0,
    mask_threshold: float = 0.5,
) -> dict[str, float]:
    """Physics metrics for **one** sample (no batch dimension).

    Parameters
    ----------
    query_masks : [Q, S, h, w] raw mask logits (pre-sigmoid).
    query_phys_mu : [Q, P] property means in z-space (``SegVGGTPrediction.query_phys_mu``).
    instance_mask : [S, H, W] int64, multi-view-consistent instance ids.
    target : the scene's :class:`PhysGMTarget` (z-space LUTs).
    valid_mask : [S, H, W] bool; False pixels are ignored entirely.

    Returns
    -------
    dict of plain floats.  Empty when nothing could be scored (no GT instance
    carries a label, or no query matched one) -- an empty dict is skipped by the
    batch wrapper rather than averaged in as a zero.
    """
    q, s, h, w = query_masks.shape
    device = query_masks.device
    p = query_phys_mu.shape[-1]

    ids = torch.unique(instance_mask)
    ids = ids[ids != ignore_id]
    if ids.numel() == 0:
        return {}

    masks_full = (instance_mask[None] == ids[:, None, None, None]).float()
    gt = _downsample_binary(masks_full, (h, w)).reshape(ids.numel(), s * h * w)
    # Same contract as instance_metrics / LossSegVGGT._build_targets.
    nonempty = gt.sum(dim=1) > 0
    gt, ids = gt[nonempty], ids[nonempty]

    # Only instances the teacher actually labelled can be scored.
    valid = target.valid.to(device)
    in_range = ids < valid.shape[0]
    labelled = torch.zeros_like(in_range)
    labelled[in_range] = valid[ids[in_range]]
    gt, ids = gt[labelled], ids[labelled]
    if ids.numel() == 0:
        return {}

    keep = None
    if valid_mask is not None:
        if valid_mask.shape[-1] == 1 and valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(-1)
        keep = _downsample_binary(valid_mask.to(torch.bool).float()[None], (h, w)).reshape(-1)

    pred_masks = (query_masks.reshape(q, s * h * w).float().sigmoid() > mask_threshold).float()
    iou = _pairwise_iou(pred_masks, gt, keep)                      # [Q, K]
    q_idx, g_idx = _optimal_assignment(iou)
    if q_idx.numel() == 0:
        return {}
    q_idx, g_idx = q_idx.to(device), g_idx.to(device)

    z_pred = query_phys_mu[q_idx].float()                          # [M, P]
    matched_ids = ids[g_idx]
    z_gt = target.value_lut.to(device)[matched_ids].float()        # [M, P]
    std = torch.tensor([spec.std for spec in PHYSGM_NORMALIZATION], device=device)[:p]

    metrics: dict[str, float] = {"phys_n_matched": float(z_pred.shape[0])}
    names = target.property_names
    err = (z_pred - z_gt).abs() * std
    # Constant baseline: predict the training mean, which is z = 0 by definition.
    err_const = z_gt.abs() * std
    err_clut = None
    if target.class_mean_lut is not None:
        z_clut = target.class_mean_lut.to(device)[matched_ids].float()
        err_clut = (z_clut - z_gt).abs() * std

    for i, prop in enumerate(names[:p]):
        base = _metric_name(prop, i)
        metrics[base] = float(err[:, i].mean())
        metrics[f"{base}_const"] = float(err_const[:, i].mean())
        if err_clut is not None:
            metrics[f"{base}_clut"] = float(err_clut[:, i].mean())
    return metrics


@torch.no_grad()
def compute_physics_metrics_batch(
    query_masks: Tensor,
    query_phys_mu: Tensor,
    instance_mask: Tensor,
    targets: list[PhysGMTarget],
    valid_mask: Tensor | None = None,
    **kwargs,
) -> dict[str, float]:
    """Batched wrapper: averages :func:`compute_physics_metrics` over B samples.

    Samples that could not be scored contribute nothing.  Returns ``{}`` when no
    sample could be scored, so the caller logs nothing rather than a fake zero.
    """
    rows: list[dict[str, float]] = []
    for i in range(query_masks.shape[0]):
        m = compute_physics_metrics(
            query_masks[i],
            query_phys_mu[i],
            instance_mask[i],
            targets[i],
            None if valid_mask is None else valid_mask[i],
            **kwargs,
        )
        if m:
            rows.append(m)
    if not rows:
        return {}
    # Scenes differ in how many instances matched; keys present in only some
    # rows (the class-lookup row) are averaged over the rows that have them.
    keys = {k for r in rows for k in r}
    return {
        k: sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r)
        for k in keys
    }

"""Object-query decode + physics readout, shared by the SegVGGT entry points.

Lifted verbatim out of ``scripts/segvggt_infer.py`` so that
``scripts/trace_instance_to_gaussians.py``'s ``segvggt`` feature source consumes the
*same* thresholds and the same ``query_idx`` join key. Two copies would drift.

``decode_instances`` mirrors ``eval/instance_eval_common.predict_by_feat_instance``;
``report_physics`` denormalises the per-query physics readout back to SI.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("segvggt_decode")


def decode_instances(
    cls_logits: torch.Tensor,   # [Q, C+1]
    mask_logits: torch.Tensor,  # [Q, V*h*w]
    mask_thr: float = 0.4,
    topk: int = 600,
    npoint_thr: int = 200,
    score_thr: float = 0.25,
    class_agnostic: bool = False,
):
    """Return (binary_masks [N, V*h*w], scores [N], labels [N], query_idx [N]).

    ``query_idx`` maps each surviving instance back to the object query that produced
    it, so per-query readouts (e.g. physics mu/var) can be lined up with the masks.
    """
    cls_logits = cls_logits.float().cpu()
    mask_logits = mask_logits.float().cpu()

    if class_agnostic:
        probs = F.softmax(cls_logits, dim=-1)
        scores = 1.0 - probs[:, -1]  # 1 - P(no-match)
        topk = min(topk, scores.shape[0])
        scores, idx = scores.topk(topk, sorted=False)
        m = mask_logits[idx]
        labels = torch.zeros_like(scores, dtype=torch.long)
    else:
        # Per-(query, class) scores, excluding the trailing no-match channel.
        n_classes = cls_logits.shape[1] - 1
        scores = F.softmax(cls_logits, dim=-1)[:, :-1]
        labels = (
            torch.arange(n_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(len(cls_logits), 1)
            .flatten(0, 1)
        )
        scores, flat_idx = scores.flatten(0, 1).topk(min(topk, scores.numel()), sorted=False)
        labels = labels[flat_idx]
        idx = torch.div(flat_idx, n_classes, rounding_mode="floor")
        m = mask_logits[idx]

    m_sig = m.sigmoid()
    mask_scores = (m_sig * (m > 0)).sum(1) / ((m > 0).sum(1) + 1e-6)
    scores = scores * mask_scores

    binary = m_sig > mask_thr
    keep = scores > score_thr
    scores, binary, labels, idx = scores[keep], binary[keep], labels[keep], idx[keep]
    keep = binary.sum(1) > npoint_thr
    scores, binary, labels, idx = scores[keep], binary[keep], labels[keep], idx[keep]
    order = scores.argsort(descending=True)
    return binary[order], scores[order], labels[order], idx[order]


def report_physics(pred, query_idx: torch.Tensor, scores: torch.Tensor, out_dir: Path):
    """Print / dump the per-object physics readout for the surviving instances.

    This is the end product of the physics-on-queries route: because the properties
    are decoded from the object queries rather than pooled with a GT mask, they are
    available here at inference on a scene with no annotation at all.

    Values are converted back to SI (density kg/m3, Young's modulus Pa, Poisson ratio)
    with the dataset's own inverse transform, so nothing about the normalisation is
    re-derived here. No-op when the encoder ran without ``phys_scheme``.
    """
    mu = getattr(pred, "query_phys_mu", None)
    var = getattr(pred, "query_phys_var", None)
    if mu is None or var is None:
        logger.info("no physics readout in this checkpoint "
                    "(encoder cfg phys_scheme is null) -- skipping")
        return
    if len(query_idx) == 0:
        logger.info("no instances survived thresholding; nothing to report")
        return

    from src.dataset.physics.parsers import physgm_denormalize

    names = getattr(pred, "property_names", None) or ("density", "youngs_modulus",
                                                      "poisson_ratio")
    mu_sel = mu[0].float().cpu()[query_idx]                 # [N, P] model space
    var_sel = var[0].float().cpu()[query_idx]
    si = physgm_denormalize(mu_sel)                          # [N, P] SI units

    body = format_physics_table(si, var_sel, scores, names)
    logger.info("per-object physics (SI units, +- is the model-space std):\n%s", body)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "physics.npz",
        query_idx=query_idx.cpu().numpy(),
        scores=scores.cpu().numpy(),
        mu_model_space=mu_sel.numpy(),
        var_model_space=var_sel.numpy(),
        value_si=si.numpy(),
        property_names=np.array(names),
    )
    logger.info("wrote %s", out_dir / "physics.npz")


def format_physics_table(si, var_model, scores, names, max_rows: int = 40) -> str:
    """Render the per-object physics readout as a fixed-width table.

    ``si`` is in SI units (``physgm_denormalize`` output); ``var_model`` stays in
    model space, so the reported ``+-`` is the model-space std -- the same convention
    both entry points print, which is why this lives here rather than in either one.
    """
    header = f"{'inst':>4} {'score':>7} " + " ".join(f"{n:>18}" for n in names)
    lines = [header, "-" * len(header)]
    n = int(si.shape[0])
    for i in range(min(n, max_rows)):
        cells = " ".join(
            f"{float(si[i, p]):>10.4g}+-{float(var_model[i, p]) ** 0.5:>5.2f}"
            for p in range(si.shape[1])
        )
        lines.append(f"{i:>4} {float(scores[i]):>7.3f} {cells}")
    if n > max_rows:
        lines.append(f"... {n - max_rows} more rows")
    return "\n".join(lines)

"""Bipartite (Hungarian) matcher for SegVGGT instance training.

Reimplements the DETR / Mask2Former set-matching adapted to SegVGGT's *multi-view*
masks, following the paper (Sec. 3.3, Eq. 5 & 8).  The official SegVGGT release ships
no training code, so this is a from-scratch reproduction of the described cost.

Key idea from the paper: flattening a query's multi-view mask
``M_j in [0,1]^{N x H x W}`` into a 1-D sequence ``m_j in [0,1]^{N H W}`` makes it
mathematically equivalent to a 3-D point-cloud mask.  Applying the same flattening to
the multi-view-consistent 2-D GT masks lets us inherit the point-cloud-segmentation
matching cost (BCE + Dice) verbatim.  On top of that, SegVGGT adds a Frame-level
Attention Distribution Alignment (FADA) cost based on the Jensen-Shannon divergence
between each query's per-frame attention distribution and the GT instance visibility.

Pair-wise cost (Eq. 5), one query ``j`` vs one GT instance ``k``::

    C_{j,k} = -lambda_cls * c_{j,c_k}
              + lambda_mask * ( BCE(m_j, m_k^gt) + Dice(m_j, m_k^gt) )
              + lambda_js  * C^{js}_{j,k}

where ``C^{js}_{j,k} = 1/(L*N) * sum_l JS(p_k^gt || p_hat_j^(l))`` (Eq. 8).

This module has **no learnable parameters** and no repo dependencies (torch + scipy
only), so it lives in the loss layer and is unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

try:  # scipy is the standard optimal-assignment backend (as in DETR / Mask2Former)
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback keeps the matcher importable
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Pairwise cost primitives (all return a [Q, K] cost matrix)
# --------------------------------------------------------------------------- #
def batch_sigmoid_ce_cost(
    mask_logits: Tensor, gt_masks: Tensor, keep: Tensor | None = None
) -> Tensor:
    """Binary-cross-entropy matching cost between every prediction and GT.

    Parameters
    ----------
    mask_logits : Tensor [Q, N]   raw (pre-sigmoid) query .dot. feature scores
    gt_masks    : Tensor [K, N]   binary {0,1} flattened multi-view GT masks
    keep        : Tensor [N] or None   1 = supervised pixel, 0 = ignored.  An ignored
        pixel counts as neither positive nor negative.  Needed wherever the label is
        *unknown* rather than known-empty: folding those into the negatives would
        actively teach the model "there is no object here".

    Returns
    -------
    Tensor [Q, K]  mean-per-element BCE cost over the kept pixels.
    """
    n = mask_logits.shape[1] if keep is None else keep.sum().clamp_min(1.0)
    pos = F.binary_cross_entropy_with_logits(
        mask_logits, torch.ones_like(mask_logits), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        mask_logits, torch.zeros_like(mask_logits), reduction="none"
    )
    if keep is not None:
        pos = pos * keep
        neg = neg * keep
    # cost[q, k] = mean_n( pos[q,n] * gt[k,n] + neg[q,n] * (1 - gt[k,n]) )
    cost = pos @ gt_masks.transpose(0, 1) + neg @ (1.0 - gt_masks).transpose(0, 1)
    return cost / n


def batch_dice_cost(
    mask_logits: Tensor,
    gt_masks: Tensor,
    eps: float = 1.0,
    keep: Tensor | None = None,
) -> Tensor:
    """Dice matching cost with Laplace smoothing (V-Net style, eps=1).

    mask_logits : [Q, N] logits ; gt_masks : [K, N] binary.  Returns [Q, K].
    ``keep`` [N] drops ignored pixels from both numerator and denominator.
    """
    probs = mask_logits.sigmoid()
    if keep is not None:
        probs = probs * keep
        gt_masks = gt_masks * keep
    numerator = 2.0 * (probs @ gt_masks.transpose(0, 1))          # [Q, K]
    denominator = probs.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]  # [Q, K]
    return 1.0 - (numerator + eps) / (denominator + eps)


def _kl(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """KL(p || q) over the last dim, with 0*log0 = 0 handled by the eps floor."""
    p = p.clamp_min(0.0)
    q = q.clamp_min(eps)
    return (p * (p.clamp_min(eps).log() - q.log())).sum(-1)


def js_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric, bounded Jensen-Shannon divergence over the last dim.

    ``p`` and ``q`` are probability distributions (sum to 1 over the last dim).
    Broadcasts, so shapes like ``p:[...,1,S]`` vs ``q:[...,K,S]`` work.
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)


def batch_js_frame_cost(
    pred_frame_attn: Tensor,  # [L, Q, S] per-frame attention distributions (sum_S = 1)
    gt_visibility: Tensor,    # [K, S]    per-frame GT visibility distributions (sum_S = 1)
) -> Tensor:
    """FADA matching cost C^{js}_{j,k} = 1/(L*N) sum_l JS(p_k^gt || p_hat_j^(l)).

    Returns [Q, K].  N = number of frames S (folded in as the paper's 1/N factor).
    """
    L, Q, S = pred_frame_attn.shape
    p = gt_visibility[None, None, :, :]          # [1, 1, K, S]
    q = pred_frame_attn[:, :, None, :]           # [L, Q, 1, S]
    js = js_divergence(p, q)                      # [L, Q, K]
    return js.mean(dim=0) / S                     # [Q, K]  (mean over L, /N)


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #
@dataclass
class MatcherWeights:
    cost_cls: float = 0.5    # lambda_cls
    cost_mask: float = 1.0   # lambda_mask (shared by BCE + Dice)
    cost_js: float = 0.5     # lambda_js  (0 disables the FADA cost term, Table 3 row 2)


class HungarianMatcher:
    """Optimal query<->GT assignment for one sample (no batching, no grad)."""

    def __init__(self, weights: MatcherWeights) -> None:
        self.w = weights

    @torch.no_grad()
    def match_one(
        self,
        cls_prob: Tensor,           # [Q, C+1] softmax probabilities
        mask_logits: Tensor,        # [Q, N]   flattened multi-view mask logits
        gt_classes: Tensor,         # [K]      long, in [0, C-1]
        gt_masks: Tensor,           # [K, N]   binary flattened multi-view GT masks
        pred_frame_attn: Tensor | None = None,  # [L, Q, S]
        gt_visibility: Tensor | None = None,    # [K, S]
        keep: Tensor | None = None,             # [N] 1 = supervised, 0 = ignored
    ) -> tuple[Tensor, Tensor]:
        """Return ``(query_idx, gt_idx)`` LongTensors of the matched pairs.

        ``keep`` must be the same pixel mask the loss uses, otherwise matching and
        supervision disagree about which pixels count.
        """
        device = mask_logits.device
        k = gt_classes.shape[0]
        if k == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty

        # -- classification cost: -P(query = c_k) ------------------------------
        cost_cls = -cls_prob[:, gt_classes]                       # [Q, K]

        # -- mask cost: BCE + Dice over flattened NHW --------------------------
        cost_mask = batch_sigmoid_ce_cost(
            mask_logits, gt_masks, keep
        ) + batch_dice_cost(mask_logits, gt_masks, keep=keep)

        cost = self.w.cost_cls * cost_cls + self.w.cost_mask * cost_mask

        # -- FADA frame-attention cost (Eq. 8) ---------------------------------
        if (
            self.w.cost_js > 0.0
            and pred_frame_attn is not None
            and gt_visibility is not None
        ):
            cost = cost + self.w.cost_js * batch_js_frame_cost(
                pred_frame_attn, gt_visibility
            )

        cost = torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=-1e4)
        cost_cpu = cost.detach().cpu()

        if _HAS_SCIPY:
            row, col = linear_sum_assignment(cost_cpu.numpy())
            query_idx = torch.as_tensor(row, dtype=torch.long, device=device)
            gt_idx = torch.as_tensor(col, dtype=torch.long, device=device)
        else:  # greedy fallback (deterministic, no external dep)
            query_idx, gt_idx = _greedy_assignment(cost_cpu, device)

        return query_idx, gt_idx


def _greedy_assignment(cost: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """Greedy min-cost assignment fallback when scipy is unavailable."""
    q, k = cost.shape
    n = min(q, k)
    used_q: set[int] = set()
    used_k: set[int] = set()
    flat = cost.reshape(-1)
    order = torch.argsort(flat)
    rows: list[int] = []
    cols: list[int] = []
    for idx in order.tolist():
        r, c = divmod(idx, k)
        if r in used_q or c in used_k:
            continue
        used_q.add(r)
        used_k.add(c)
        rows.append(r)
        cols.append(c)
        if len(rows) == n:
            break
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )

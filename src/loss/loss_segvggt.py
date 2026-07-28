"""SegVGGT end-to-end instance loss: classification + mask (BCE+Dice) + FADA (JS).

Reproduces the instance training objective of SegVGGT (paper Sec. 3.3 / 3.4), which
the official release does not ship.  After a Hungarian match between the O object
queries and the GT instances of a scene (see :mod:`segvggt_matcher`), we apply::

    L_inst = lambda_cls * L_cls + lambda_mask * (L_bce + L_dice)
    L_js   = 1/(|M|*L*N) * sum_{(j,k) in M} sum_l JS(p_k^gt || p_hat_j^(l))

and this module returns their weighted sum ``lambda_cls*L_cls +
lambda_mask*(L_bce+L_dice) + lambda_js*L_js`` (the FADA term is folded in here
because it shares the match).  The geometry terms (``L_camera``, ``L_depth``) live
in :mod:`loss_segvggt_geo`.

**Per-query physics** (``lambda_phys``, default 0 = off) is folded in for the same
reason: it supervises the object queries through *the same* Hungarian assignment, so
it cannot be a separate Loss without either recomputing the match (silently diverging
the moment the two configs' cost weights disagree) or introducing an execution-order
dependency between losses, which this repo has nowhere.  When enabled it applies
PhysGM's per-property robust Gaussian NLL + MSE (formula reused verbatim from
:mod:`loss_physgm`) to ``query_phys_mu`` / ``query_phys_var`` against the matched
instance's row of a ``PhysGMTarget`` LUT.  Note the LUT is keyed by *instance id*,
so the match's ``gt_idx`` must be mapped through ``ids`` -- see ``_build_targets``.

GT construction (all derived from the multi-view ``instance_mask``, no extra parser):
  * per-instance binary masks, area-downsampled to the prediction resolution (h, w)
    and flattened to the ``N*h*w`` sequence the paper treats as a point-cloud mask;
  * per-instance frame-visibility distribution ``p_k^gt`` (area-proportional over the
    N frames, unseen frames -> 0), used by the FADA JS term;
  * per-instance class label.  Defaults to **class-agnostic**: this repo's manifest
    datasets only carry multi-view-consistent instance ids, no semantic label.  In that
    mode the classification collapses to *objectness* by marginalising the class
    distribution, ``P(object) = sum_c P(c) = 1 - P(no-match)``, i.e. a 2-way CE over
    ``[logsumexp(foreground logits), no-match logit]``.  This keeps the pretrained
    classifier intact (no head surgery, no channel is arbitrarily repurposed), matches
    the ``1 - P(no-match)`` criterion used at inference, and leaves the foreground
    channels' semantic structure available for later per-query readouts.
    If a per-pixel ``instance_semantic`` map is supplied via ``depth_dict`` *and*
    ``class_agnostic=False``, it is majority-voted per instance for class-aware targets.

Inputs (provided by :class:`SegVGGTWrapper` through ``depth_dict``):
  - depth_dict['segvggt_prediction']: :class:`SegVGGTPrediction`
      * query_masks        [B, Q, S, h, w]  (raw logits, pre-sigmoid)
      * query_class_logits [B, Q, C+1]      (last channel = no-match; DETR: no-object)
      * attn_frame_mean    [L, B, Q, S]     (per-frame attn mass; renormalised here)
      * query_phys_mu/var  [B, Q, P]        (optional, lambda_phys > 0)
  - depth_dict['physgm_target']:       list[PhysGMTarget] (optional, lambda_phys > 0)
  - depth_dict['instance_mask']:       Int64 [B, S, H, W]
  - depth_dict['instance_valid_mask']: Bool  [B, S, H, W]  (optional).  Invalid pixels
    are *ignored* by the mask loss, not turned into negatives -- see ``unlabeled_as_ignore``.
  - depth_dict['instance_semantic']:   Int64 [B, S, H, W]  (optional, class-aware)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from .loss import Loss
from .loss_physgm import robust_gaussian_nll
from .segvggt_matcher import (
    HungarianMatcher,
    MatcherWeights,
    batch_dice_cost,
    js_divergence,
)

logger = logging.getLogger(__name__)


@dataclass
class LossSegVGGTCfg:
    weight: float = 1.0
    # loss-term weights (paper A.4: lambda_cls=0.5, lambda_mask=1.0, lambda_js=0.5)
    lambda_cls: float = 0.5
    lambda_mask: float = 1.0
    lambda_js: float = 0.5
    # matching-cost weights (paper: "exactly match their corresponding loss counterparts")
    cost_cls: float = 0.5
    cost_mask: float = 1.0
    cost_js: float = 0.5
    # DETR down-weights the no-match class in the classification CE
    # (config field keeps the DETR name ``no_object_weight``).
    no_object_weight: float = 0.1
    ignore_id: int = 0          # instance id that does not define an instance
    # Drop id == ignore_id pixels from the *mask* BCE/Dice entirely (False = keep them).
    # NOTE: this switch aims at the wrong place and defaults off.  A matched query's mask
    # target is (inst_mask == id_k): every id-0 pixel is a truthful negative ("not part
    # of instance k"), so the mask loss never lies even when id 0 hides an unannotated
    # object.  The harm from under-labelling is on the *classification* side instead -- a
    # query that latched onto that object goes unmatched and is supervised as no-match.
    # Turning this on therefore deletes real negative signal (masks lose the pressure to
    # stay tight and grow), without addressing the actual issue.  No known case needs it.
    unlabeled_as_ignore: bool = False
    # Collapse classification to objectness (see module docstring).  True for this
    # repo's manifest datasets, which carry instance ids but no semantic labels.
    class_agnostic: bool = True
    dice_eps: float = 1.0        # Laplace smoothing for Dice
    # Per-query physics property regression (PhysGM formula), 0 = off.  Requires the
    # encoder to run QueryPhysGMReadout (cfg `phys_scheme: query_physgm`) and the
    # dataset to supply a PhysGMTarget.  Folded into this loss rather than living in
    # its own module because it must consume *the same* Hungarian match as the masks --
    # exactly the reason the FADA term is folded in here too (see module docstring).
    lambda_phys: float = 0.0
    phys_mse_weight: float = 1.0  # PhysGM: E = NLL + mse_weight * MSE, per property


@dataclass
class LossSegVGGTCfgWrapper:
    segvggt: LossSegVGGTCfg


class LossSegVGGT(Loss[LossSegVGGTCfg, LossSegVGGTCfgWrapper]):
    """Hungarian-matched classification + mask + FADA loss for SegVGGT."""

    def __init__(self, cfg: LossSegVGGTCfgWrapper) -> None:
        super().__init__(cfg)
        self.matcher = HungarianMatcher(
            MatcherWeights(
                cost_cls=self.cfg.cost_cls,
                cost_mask=self.cfg.cost_mask,
                cost_js=self.cfg.cost_js,
            )
        )

    # ------------------------------------------------------------------ #
    # GT builders (pure tensor ops, no learnable params)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _downsample_masks(binary: Tensor, hw: tuple[int, int]) -> Tensor:
        """[K, S, H, W] binary -> [K, S, h, w] binary via area resample + 0.5 thresh."""
        k, s, H, W = binary.shape
        h, w = hw
        if (H, W) == (h, w):
            return binary
        x = binary.reshape(k * s, 1, H, W).float()
        x = F.interpolate(x, size=(h, w), mode="area")
        return (x.reshape(k, s, h, w) > 0.5).float()

    def _build_targets(
        self,
        inst_mask_b: Tensor,          # [S, H, W] int64
        semantic_b: Tensor | None,    # [S, H, W] int64 or None
        hw: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (gt_classes [K], gt_masks [K, S, h, w], gt_visibility [K, S], ids [K]).

        ``ids`` are the *actual* instance ids, in the same row order as the other three.
        The matcher returns row indices into this ordering, so any supervision keyed by
        instance id (e.g. the per-instance physics LUTs) needs ``ids[gt_idx]`` -- the
        raw ``gt_idx`` is a row number, not an id.

        Instances that vanish under ``_downsample_masks`` (area + 0.5) are dropped so
        Hungarian never sees an empty positive mask paired with an object class target.
        """
        s, H, W = inst_mask_b.shape
        ids = torch.unique(inst_mask_b)
        ids = ids[ids != self.cfg.ignore_id]
        device = inst_mask_b.device
        h, w = hw
        if ids.numel() == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, s, h, w, device=device),
                torch.zeros(0, s, device=device),
                ids,
            )

        masks_full = (inst_mask_b[None] == ids[:, None, None, None]).float()  # [K,S,H,W]
        gt_masks = self._downsample_masks(masks_full, hw)        # [K, S, h, w]
        # Pred-resolution emptiness: full-res unique can still yield all-zero masks
        # after area+0.5 (tiny / thin instances). Drop them before matching.
        keep = gt_masks.reshape(ids.numel(), -1).sum(dim=1) > 0
        ids = ids[keep]
        masks_full = masks_full[keep]
        gt_masks = gt_masks[keep]
        if ids.numel() == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, s, h, w, device=device),
                torch.zeros(0, s, device=device),
                ids,
            )

        # frame-visibility distribution p_k^gt (area-proportional over S frames)
        per_frame = masks_full.sum(dim=(-1, -2))                 # [K, S]
        visibility = per_frame / per_frame.sum(dim=1, keepdim=True).clamp_min(1e-6)

        if self.cfg.class_agnostic or semantic_b is None:
            # class 0 = the "object" column of the marginalised objectness distribution
            gt_classes = torch.zeros(ids.numel(), dtype=torch.long, device=device)
        else:
            gt_classes = torch.empty(ids.numel(), dtype=torch.long, device=device)
            for i, inst_id in enumerate(ids):
                sem_vals = semantic_b[inst_mask_b == inst_id]
                sem_vals = sem_vals[sem_vals >= 0]
                if sem_vals.numel() == 0:
                    gt_classes[i] = 0
                else:
                    gt_classes[i] = torch.bincount(sem_vals).argmax()
        return gt_classes, gt_masks, visibility, ids

    # ------------------------------------------------------------------ #
    # Loss interface
    # ------------------------------------------------------------------ #
    def forward(
        self,
        prediction,
        batch: BatchedExample,
        gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.extra_logs: dict[str, Tensor] = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _zero():
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        def _zero_with_graph(pred=None) -> Tensor:
            """Scalar 0 that stays on the autograd graph when predictions exist.

            Bare ``torch.tensor(0.)`` has no grad_fn; with ``lambda_phys`` as the
            sole live term (``segvggt_physgm``: seg λ=0, only ``query_physgm``
            trains), an empty-phys-supervision step would then crash on backward.
            Prefer hanging on ``query_phys_mu/var``; fall back to mask/cls tensors
            or a requires_grad leaf (same pattern as ``LossPhysGM._zero``).
            """
            terms: list[Tensor] = []
            if pred is not None:
                mu = getattr(pred, "query_phys_mu", None)
                var = getattr(pred, "query_phys_var", None)
                if mu is not None:
                    terms.append(mu.float().sum())
                if var is not None:
                    terms.append(var.float().sum())
                if not terms:
                    if getattr(pred, "query_masks", None) is not None:
                        terms.append(pred.query_masks.float().sum())
                    if getattr(pred, "query_class_logits", None) is not None:
                        terms.append(pred.query_class_logits.float().sum())
            if terms:
                acc = terms[0]
                for t in terms[1:]:
                    acc = acc + t
                return acc * 0.0
            return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

        if depth_dict is None:
            return _zero_with_graph()
        pred = depth_dict.get("segvggt_prediction")
        inst_mask = depth_dict.get("instance_mask")
        if pred is None or inst_mask is None:
            return _zero_with_graph(pred)
        if pred.query_masks is None or pred.query_class_logits is None:
            return _zero_with_graph(pred)

        query_masks = pred.query_masks                    # [B, Q, S, h, w]
        query_cls = pred.query_class_logits               # [B, Q, C+1]
        attn = pred.attn_frame_mean                        # [L, B, Q, S] or None
        device = query_masks.device

        B, Q, S, h, w = query_masks.shape

        # Class-agnostic: marginalise the class distribution into a binary objectness
        # one, [logsumexp(foreground), no-match].  Mathematically P(object) = 1 -
        # P(no-match), so the pretrained classifier keeps its meaning as-is instead of
        # having channel 0 repurposed from a semantic class into "any object".
        if self.cfg.class_agnostic:
            query_cls = torch.stack(
                [query_cls[..., :-1].logsumexp(dim=-1), query_cls[..., -1]], dim=-1
            )                                              # [B, Q, 2]
        n_cls = query_cls.shape[-1]
        no_obj = n_cls - 1

        # Invalid pixels carry an untrustworthy id: they must not define instances,
        # and they must not be supervised either way.  Zeroing them into the mask (as
        # the contrastive IGGT losses do) is harmless for a pull/push objective but
        # wrong for set prediction -- it would assert "no object here".  So zero them
        # for target *construction* and drop them from the mask loss via `keep`.
        valid_mask = depth_dict.get("instance_valid_mask")
        keep_full = None  # [S, H, W] float, 1 = supervised pixel
        if valid_mask is not None:
            if valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            valid_mask = valid_mask.to(torch.bool)
            inst_mask = inst_mask * valid_mask.to(inst_mask.dtype)
            keep_full = valid_mask.float()
        if self.cfg.unlabeled_as_ignore:
            unlabeled = (inst_mask == self.cfg.ignore_id).float()
            keep_full = (1.0 - unlabeled) if keep_full is None else keep_full * (1.0 - unlabeled)
        semantic = depth_dict.get("instance_semantic")

        # class-weight vector: down-weight no-match (DETR convention / no_object_weight)
        cls_weight = torch.ones(n_cls, device=device)
        cls_weight[no_obj] = self.cfg.no_object_weight

        # ----- per-query physics (optional) -----
        # Only active with a readout on the encoder *and* a PhysGMTarget from the
        # dataset; otherwise this whole branch stays dormant and costs nothing.
        phys_mu = getattr(pred, "query_phys_mu", None)
        phys_var = getattr(pred, "query_phys_var", None)
        phys_targets = depth_dict.get("physgm_target")
        phys_on = (
            self.cfg.lambda_phys > 0.0
            and phys_mu is not None
            and phys_var is not None
            and phys_targets is not None
        )
        if phys_on and not isinstance(phys_targets, list):
            phys_targets = [phys_targets]
        phys_mu_chunks: list[Tensor] = []
        phys_var_chunks: list[Tensor] = []
        phys_gt_chunks: list[Tensor] = []

        total_cls = _zero()
        total_bce = _zero()
        total_dice = _zero()
        total_js = _zero()
        n_masks = 0  # matched pairs across the batch (mask/js normaliser)

        for b in range(B):
            gt_classes, gt_masks, visibility, gt_ids = self._build_targets(
                inst_mask[b], None if semantic is None else semantic[b], (h, w)
            )
            k = gt_classes.shape[0]

            cls_logits_b = query_cls[b].float()                    # [Q, C+1]
            cls_prob_b = cls_logits_b.softmax(-1)
            mask_logits_b = query_masks[b].reshape(Q, S * h * w).float()  # [Q, N]

            # per-pixel supervision mask, same resolution/flattening as the masks
            keep_b = None
            if keep_full is not None:
                keep_b = self._downsample_masks(keep_full[b][None], (h, w)).reshape(-1)

            pred_attn_b = None
            if attn is not None:
                a = attn[:, b].float()                              # [L, Q, S]
                pred_attn_b = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            # ----- classification target (default = no-match) -----
            tgt_classes = torch.full((Q,), no_obj, dtype=torch.long, device=device)

            if k > 0:
                gt_masks_flat = gt_masks.reshape(k, S * h * w)     # [K, N]
                q_idx, g_idx = self.matcher.match_one(
                    cls_prob_b, mask_logits_b, gt_classes, gt_masks_flat,
                    pred_attn_b, visibility, keep_b,
                )
                tgt_classes[q_idx] = gt_classes[g_idx]

                # ----- mask losses on matched pairs (ignored pixels excluded) -----
                m_logits = mask_logits_b[q_idx]                    # [M, N]
                m_gt = gt_masks_flat[g_idx]                        # [M, N]
                bce_el = F.binary_cross_entropy_with_logits(
                    m_logits, m_gt, reduction="none"
                )
                if keep_b is None:
                    bce_mean = bce_el.mean()
                else:
                    bce_mean = (bce_el * keep_b).sum() / (
                        keep_b.sum().clamp_min(1.0) * max(q_idx.numel(), 1)
                    )
                total_bce = total_bce + bce_mean * q_idx.numel()
                # Dice loss (1 - dice) on the matched diagonal
                dice_pairs = batch_dice_cost(
                    m_logits, m_gt, self.cfg.dice_eps, keep=keep_b
                ).diagonal()
                total_dice = total_dice + dice_pairs.sum()

                # ----- FADA JS regularisation on matched pairs -----
                if pred_attn_b is not None:
                    p_gt = visibility[g_idx]                       # [M, S]
                    p_hat = pred_attn_b[:, q_idx, :]               # [L, M, S]
                    js = js_divergence(p_gt[None], p_hat)          # [L, M]
                    total_js = total_js + (js.mean(dim=0) / S).sum()

                # ----- per-query physics on the SAME matched pairs -----
                # gt_idx are row numbers; gt_ids[g_idx] turns them into the instance
                # ids the PhysGMTarget LUTs are indexed by.
                if phys_on and b < len(phys_targets):
                    tgt_b = phys_targets[b]
                    value_lut = tgt_b.value_lut.to(device)
                    valid_lut = tgt_b.valid.to(device)
                    matched_ids = gt_ids[g_idx].long()
                    in_range = matched_ids < valid_lut.shape[0]
                    if bool(in_range.any()):
                        ids_ok = matched_ids[in_range]
                        labelled = valid_lut[ids_ok]
                        if bool(labelled.any()):
                            sel = q_idx[in_range][labelled]
                            phys_mu_chunks.append(phys_mu[b][sel])
                            phys_var_chunks.append(phys_var[b][sel])
                            phys_gt_chunks.append(value_lut[ids_ok[labelled]])

                n_masks += q_idx.numel()

            # ----- classification CE over all queries (matched + no-match) -----
            total_cls = total_cls + F.cross_entropy(
                cls_logits_b, tgt_classes, weight=cls_weight, reduction="mean"
            )

        # normalise
        total_cls = total_cls / max(B, 1)
        norm = max(n_masks, 1)
        total_bce = total_bce / norm
        total_dice = total_dice / norm
        total_js = total_js / norm

        # ----- physics: PhysGM formula over the matched, labelled instances -----
        # Same per-property NLL + MSE on z-scored targets as LossPhysGM, so the numbers
        # are directly comparable with the IGGT physgm route.
        # Start on-graph so an empty-chunk step still backward under physgm-only recipes.
        total_phys = _zero_with_graph(pred)
        n_phys = 0
        if phys_mu_chunks:
            mu = torch.cat(phys_mu_chunks, dim=0).float()      # [M, P]
            var = torch.cat(phys_var_chunks, dim=0).float()
            gt = torch.cat(phys_gt_chunks, dim=0).float()
            n_phys = mu.shape[0]
            prop_names = (
                getattr(pred, "property_names", None)
                or tuple(f"prop_{i}" for i in range(mu.shape[1]))
            )
            for p_i, name in enumerate(prop_names):
                nll = robust_gaussian_nll(mu[:, p_i], var[:, p_i], gt[:, p_i])
                mse = F.mse_loss(mu[:, p_i], gt[:, p_i])
                total_phys = total_phys + nll + self.cfg.phys_mse_weight * mse
                self.extra_logs[f"segvggt_phys_{name}_nll"] = nll.detach()
                self.extra_logs[f"segvggt_phys_{name}_mse"] = mse.detach()

        l_cls = self.cfg.lambda_cls * total_cls
        l_mask = self.cfg.lambda_mask * (total_bce + total_dice)
        l_js = self.cfg.lambda_js * total_js
        l_phys = self.cfg.lambda_phys * total_phys
        loss = self.cfg.weight * (l_cls + l_mask + l_js + l_phys)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.extra_logs.update({
            "loss_segvggt_cls": total_cls.detach(),
            "loss_segvggt_bce": total_bce.detach(),
            "loss_segvggt_dice": total_dice.detach(),
            "loss_segvggt_js": total_js.detach(),
            "loss_segvggt_num_matched": torch.tensor(float(n_masks)),
            "loss_segvggt_phys": total_phys.detach(),
            "segvggt_phys_num": torch.tensor(float(n_phys)),
        })
        if global_step % 100 == 0:
            logger.info(
                "[LossSegVGGT step=%d] cls=%.4f bce=%.4f dice=%.4f js=%.4f "
                "phys=%.4f matched=%d phys_n=%d",
                global_step, total_cls.item(), total_bce.item(),
                total_dice.item(), total_js.item(), total_phys.item(),
                n_masks, n_phys,
            )
        return loss

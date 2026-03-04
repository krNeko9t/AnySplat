"""Discriminative instance embedding loss (IGGT-style).

Implements the discriminative loss from De Brabandere et al. (2017):
  * Pull loss: penalise instance pixels whose embeddings deviate from the
    instance mean beyond ``delta_v``.
  * Push loss: penalise pairs of instance means that are closer than
    ``2 * delta_d``.

When ``multi_view`` is False (default), operates per-image independently.
When ``multi_view`` is True, all views of the same batch item are merged
so that per-instance means are computed across views, providing implicit
cross-view consistency (same physical instance in different views is pulled
toward a shared global mean).

Optional step-based schedule (non-invasive): set ``multi_view_step`` to
enable multi_view only when global_step >= N; set ``soft_step`` and
``soft_ramp_steps`` to ramp ``soft_push_weight`` / ``soft_pull_weight``
from 0 starting at step M. Use 0 for "from step 0" (legacy behaviour).

Expected inputs (provided via ``depth_dict`` by the training loop):
  - depth_dict['instance_feat_map']: Float[Tensor] shaped [B, V, N, H, W]
  - depth_dict['instance_mask']:     Int64[Tensor] shaped [B, V, H, W]
  - (optional) depth_dict['instance_valid_mask']: Bool[Tensor] shaped [B, V, H, W]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss

logger = logging.getLogger(__name__)


@dataclass
class LossDiscCfg:
    weight: float = 1.0
    delta_v: float = 0.5
    delta_d: float = 1.5
    ignore_id: int = 0
    # --- new optional knobs (all default to legacy behaviour) ---
    multi_view: bool = False
    alpha_var: float = 1.0
    alpha_dist: float = 1.0
    soft_push_weight: float = 0.0
    soft_pull_weight: float = 0.0
    # --- step-based schedule (0 = from step 0, no delay) ---
    multi_view_step: int = 0   # enable multi_view only when global_step >= this
    soft_step: int = 0         # start ramping soft_* from 0 at this step
    soft_ramp_steps: int = 5000  # ramp soft_* to target over this many steps


@dataclass
class LossDiscCfgWrapper:
    disc: LossDiscCfg


class LossDisc(Loss[LossDiscCfg, LossDiscCfgWrapper]):
    """Discriminative instance segmentation loss (IGGT-style)."""

    def __init__(self, cfg: LossDiscCfgWrapper) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Core discriminative loss.  Accepts any [C, *spatial] embedding
    # with a matching [*spatial] label map (works for single-view
    # [C, H, W] or multi-view merged [C, V*H*W]).
    # l_var = pull loss; l_dist = push loss
    # ------------------------------------------------------------------

    @staticmethod
    def _discriminative_loss(
        embedding: Tensor,
        seg_gt: Tensor,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        ignore_id: int = 0,
        alpha_var: float = 1.0,
        alpha_dist: float = 1.0,
        soft_push_weight: float = 0.0,
        soft_pull_weight: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute discriminative loss for one sample.

        Parameters
        ----------
        embedding : Tensor [C, N]  (N = H*W for single view, V*H*W for merged)
        seg_gt : Tensor [N]  (integer instance ids)
        delta_v / delta_d : hinge margins (same semantics as De Brabandere 2017)
        alpha_var / alpha_dist : independent weights for pull / push terms
        soft_push_weight : weight for a continuous exp-based push (0 = off)
        soft_pull_weight : weight for a continuous mean-distance pull (0 = off)

        Returns
        -------
        (total_loss, l_var_weighted, l_dist_weighted)  — all scalars
        """
        seg_flat = seg_gt.reshape(-1)
        emb_flat = embedding.reshape(embedding.shape[0], -1)  # [C, N]
        # emb_flat = F.normalize(emb_flat, dim=0)

        unique_ids = torch.unique(seg_flat)
        unique_ids = unique_ids[unique_ids != ignore_id]
        num_instances = len(unique_ids)

        _zero = torch.tensor(0.0, device=embedding.device, dtype=embedding.dtype)
        if num_instances == 0:
            return _zero, _zero, _zero

        mu_list: list[Tensor] = []
        l_var = emb_flat.new_tensor(0.0)
        l_soft_pull = emb_flat.new_tensor(0.0)

        for inst_id in unique_ids:
            mask = seg_flat == inst_id
            masked_emb = emb_flat[:, mask]          # [C, N_pixels]
            mean = masked_emb.mean(dim=1)           # [C]
            mu_list.append(mean)

            dist = torch.norm(masked_emb - mean.unsqueeze(1), dim=0)  # [N_pixels]
            l_var = l_var + torch.clamp(dist - delta_v, min=0).pow(2).mean()

            if soft_pull_weight > 0:
                l_soft_pull = l_soft_pull + dist.mean()

        l_var = l_var / num_instances
        l_soft_pull = l_soft_pull / num_instances

        l_dist = emb_flat.new_tensor(0.0)
        l_soft_push = emb_flat.new_tensor(0.0)
        if num_instances > 1:
            mu = torch.stack(mu_list)                    # [K, C]
            pair_dist = torch.norm(
                mu.unsqueeze(0) - mu.unsqueeze(1), dim=2
            )  # [K, K]
            diag_mask = torch.eye(num_instances, device=embedding.device, dtype=torch.bool)
            off_diag = pair_dist[~diag_mask]

            l_dist = torch.clamp(2 * delta_d - off_diag, min=0).pow(2).mean()

            if soft_push_weight > 0:
                l_soft_push = torch.exp(-off_diag).mean()

        total = (
            alpha_var * l_var
            + alpha_dist * l_dist
            + soft_pull_weight * l_soft_pull
            + soft_push_weight * l_soft_push
        )
        return total, alpha_var * l_var, alpha_dist * l_dist

    # ------------------------------------------------------------------
    # AnySplat Loss interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.extra_logs: dict[str, Tensor] = {}

        if depth_dict is None:
            if global_step % 100 == 0:
                logger.info(f"[LossDisc dbg step={global_step}] depth_dict is None → return 0")
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        feat_map: Tensor | None = depth_dict.get("instance_feat_map")
        inst_mask: Tensor | None = depth_dict.get("instance_mask")
        valid_mask: Tensor | None = depth_dict.get("instance_valid_mask")

        if feat_map is None or inst_mask is None:
            if global_step % 100 == 0:
                logger.info(f"[LossDisc dbg step={global_step}] feat_map={feat_map is not None}, inst_mask={inst_mask is not None} → return 0")
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        B, V, C, H, W = feat_map.shape
        device = feat_map.device

        if global_step % 100 == 0:
            logger.info(f"[LossDisc dbg step={global_step}] feat_map={feat_map.shape} inst_mask={inst_mask.shape} valid_mask={valid_mask.shape if valid_mask is not None else None}")
            logger.info(f"  inst_mask unique (pre-valid): {torch.unique(inst_mask.view(-1))[:15].tolist()}, nonzero={inst_mask.count_nonzero().item()}/{inst_mask.numel()}")
            if valid_mask is not None:
                logger.info(f"  valid_mask sum={valid_mask.sum().item()}/{valid_mask.numel()}")

        # Apply valid mask to instance labels (set invalid pixels to ignore_id).
        if valid_mask is not None:
            if valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            inst_mask = inst_mask * valid_mask.long()

        if global_step % 100 == 0:
            logger.info(f"  inst_mask unique (post-valid): {torch.unique(inst_mask.view(-1))[:15].tolist()}, nonzero={inst_mask.count_nonzero().item()}/{inst_mask.numel()}")

        # Step-based schedule: only use multi_view when global_step >= multi_view_step
        multi_view_cfg = bool(getattr(self.cfg, "multi_view", False))
        multi_view_step = int(getattr(self.cfg, "multi_view_step", 0))
        multi_view = multi_view_cfg and (global_step >= multi_view_step)

        # Step-based schedule: ramp soft weights from 0 starting at soft_step
        _soft_step = int(getattr(self.cfg, "soft_step", 0))
        _soft_ramp = int(getattr(self.cfg, "soft_ramp_steps", 5000))
        _push_t = float(getattr(self.cfg, "soft_push_weight", 0.0))
        _pull_t = float(getattr(self.cfg, "soft_pull_weight", 0.0))
        if _soft_step <= 0 or _soft_ramp <= 0:
            soft_push_w, soft_pull_w = _push_t, _pull_t
        else:
            ramp = min(1.0, max(0.0, (global_step - _soft_step) / _soft_ramp))
            soft_push_w = _push_t * ramp
            soft_pull_w = _pull_t * ramp

        loss_kwargs = dict(
            delta_v=float(self.cfg.delta_v),
            delta_d=float(self.cfg.delta_d),
            ignore_id=int(self.cfg.ignore_id),
            alpha_var=float(getattr(self.cfg, "alpha_var", 1.0)),
            alpha_dist=float(getattr(self.cfg, "alpha_dist", 1.0)),
            soft_push_weight=soft_push_w,
            soft_pull_weight=soft_pull_w,
        )

        total_loss = feat_map.new_tensor(0.0)
        total_var = feat_map.new_tensor(0.0)
        total_dist = feat_map.new_tensor(0.0)
        num_valid = 0

        if multi_view:
            # Merge all views per batch item → cross-view disc loss.
            for b in range(B):
                # feat_b = feat_map[b].reshape(C, V * H * W)
                # feat_map[b] is [V, C, H, W]
                # 先 permute 成 [C, V, H, W]，再 reshape 成 [C, V*H*W]
                feat_b = feat_map[b].permute(1, 0, 2, 3).reshape(C, V * H * W)
                mask_b = inst_mask[b].reshape(V * H * W)
                loss_b, var_b, dist_b = self._discriminative_loss(
                    feat_b, mask_b, **loss_kwargs,
                )
                if loss_b > 0:
                    total_loss = total_loss + loss_b
                    total_var = total_var + var_b
                    total_dist = total_dist + dist_b
                    num_valid += 1
        else:
            # Legacy: per-view independent disc loss.
            feat_flat = feat_map.view(B * V, C, H, W)
            mask_flat = inst_mask.view(B * V, H, W)
            for i in range(B * V):
                loss_i, var_i, dist_i = self._discriminative_loss(
                    feat_flat[i], mask_flat[i], **loss_kwargs,
                )
                if loss_i > 0:
                    total_loss = total_loss + loss_i
                    total_var = total_var + var_i
                    total_dist = total_dist + dist_i
                    num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid
            total_var = total_var / num_valid
            total_dist = total_dist / num_valid

        if global_step % 100 == 0:
            mode = "multi_view" if multi_view else "per_view"
            soft_info = f" soft_push={soft_push_w:.3f} soft_pull={soft_pull_w:.3f}" if (soft_push_w > 0 or soft_pull_w > 0) else ""
            logger.info(f"  mode={mode} num_valid={num_valid}/{B if multi_view else B*V}, total_loss={total_loss.item():.6f} (var={total_var.item():.6f} dist={total_dist.item():.6f}){soft_info}")

        loss = float(self.cfg.weight) * total_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.extra_logs = {
            "loss_disc_raw": total_loss.detach(),
            "loss_pull": total_var.detach(),
            "loss_push": total_dist.detach(),
        }

        return loss

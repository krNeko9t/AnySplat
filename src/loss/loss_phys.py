"""Physics property classification loss.

Classifies each instance in the scene into one of N physics categories
(e.g. static / rigid / soft / unknown) using dense features from the
PhysicsHead, pooled per-instance via GT masks.

Expected inputs (provided via ``depth_dict`` by the training loop):
  - depth_dict['physics_feat_map']:  Float[Tensor] shaped [B, V, C, H, W]
  - depth_dict['instance_mask']:     Int64[Tensor] shaped [B, V, H, W]
  - depth_dict['phys_label_map']:    Int64[Tensor] shaped [B, max_id+1]
                                     (1-D lookup: instance_id -> phys_class)
  - (optional) depth_dict['instance_feat_map']: Float[Tensor] shaped [B, V, D, H, W]
  - (optional) depth_dict['instance_valid_mask']: Bool[Tensor] shaped [B, V, H, W]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss utility
# ---------------------------------------------------------------------------

def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Multi-class focal loss.

    Parameters
    ----------
    logits : [N, C]
    targets : [N] (class indices)
    gamma : focusing parameter
    alpha : [C] per-class weight (optional)
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    p_t = torch.exp(-ce)
    loss = ((1 - p_t) ** gamma) * ce
    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LossPhysCfg:
    weight: float = 1.0
    num_classes: int = 4
    ignore_id: int = 0
    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: Optional[list[float]] = None
    # Classifier architecture
    phys_feat_dim: int = 32
    classifier_hidden: int = 64
    # Optional: concat instance embedding before classification
    use_inst_feat: bool = False
    inst_feat_dim: int = 8
    # Cross-view pooling
    cross_view_pool: bool = True
    # Dense auxiliary loss
    dense_aux_weight: float = 0.0


@dataclass
class LossPhysCfgWrapper:
    phys: LossPhysCfg


class LossPhys(Loss[LossPhysCfg, LossPhysCfgWrapper]):
    """Per-instance physics classification loss."""

    def __init__(self, cfg: LossPhysCfgWrapper) -> None:
        super().__init__(cfg)

        in_dim = int(self.cfg.phys_feat_dim)
        if self.cfg.use_inst_feat:
            in_dim += int(self.cfg.inst_feat_dim)

        hidden = int(self.cfg.classifier_hidden)
        n_cls = int(self.cfg.num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_cls),
        )

        if self.cfg.dense_aux_weight > 0:
            self.dense_proj = nn.Conv2d(
                int(self.cfg.phys_feat_dim), n_cls, kernel_size=1,
            )

    # ------------------------------------------------------------------

    def _pool_per_instance(
        self,
        feat_map: Tensor,
        inst_mask: Tensor,
        phys_lut: Tensor,
        inst_feat_map: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Masked average pooling over all views for each labelled instance.

        Returns (pooled_features [K, C], gt_labels [K]).
        """
        # feat_map: [V, C, H, W], inst_mask: [V, H, W], phys_lut: [max_id+1]
        V, C, H, W = feat_map.shape

        if valid_mask is not None:
            if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            inst_mask = inst_mask.clone()
            inst_mask[~valid_mask.bool()] = 0

        flat_mask = inst_mask.reshape(-1)
        unique_ids = torch.unique(flat_mask)
        unique_ids = unique_ids[unique_ids != int(self.cfg.ignore_id)]
        # Filter to only IDs that have a physics label in the lookup table
        valid_ids = unique_ids[unique_ids < phys_lut.shape[0]]
        valid_ids = valid_ids[phys_lut[valid_ids] != int(self.cfg.ignore_id)]

        if valid_ids.numel() == 0:
            device = feat_map.device
            in_dim = C + (inst_feat_map.shape[1] if inst_feat_map is not None and self.cfg.use_inst_feat else 0)
            return (
                torch.empty((0, in_dim), device=device, dtype=feat_map.dtype),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        feat_flat = feat_map.reshape(V, C, -1)  # [V, C, H*W]
        mask_flat = inst_mask.reshape(V, -1)     # [V, H*W]

        if self.cfg.use_inst_feat and inst_feat_map is not None:
            D = inst_feat_map.shape[1]
            ifeat_flat = inst_feat_map.reshape(V, D, -1)  # [V, D, H*W]

        pooled_list: list[Tensor] = []
        label_list: list[int] = []

        for inst_id in valid_ids:
            id_int = inst_id.item()
            per_view_mask = (mask_flat == id_int)  # [V, H*W]
            count = per_view_mask.sum()
            if count == 0:
                continue

            masked_feat = feat_flat * per_view_mask.unsqueeze(1).to(feat_flat.dtype)
            pooled = masked_feat.sum(dim=(0, 2)) / count.clamp(min=1).to(feat_flat.dtype)

            if self.cfg.use_inst_feat and inst_feat_map is not None:
                masked_ifeat = ifeat_flat * per_view_mask.unsqueeze(1).to(ifeat_flat.dtype)
                pooled_ifeat = masked_ifeat.sum(dim=(0, 2)) / count.clamp(min=1).to(ifeat_flat.dtype)
                pooled = torch.cat([pooled, pooled_ifeat], dim=0)

            pooled_list.append(pooled)
            label_list.append(phys_lut[id_int].item())

        if not pooled_list:
            device = feat_map.device
            in_dim = C + (inst_feat_map.shape[1] if inst_feat_map is not None and self.cfg.use_inst_feat else 0)
            return (
                torch.empty((0, in_dim), device=device, dtype=feat_map.dtype),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        features = torch.stack(pooled_list, dim=0)
        # Labels are 1-indexed (static=1, rigid=2, soft=3, unknown=4).
        # Shift to 0-indexed for cross-entropy: class 0 = static, etc.
        labels = torch.tensor(label_list, device=feat_map.device, dtype=torch.long) - 1

        return features, labels

    # ------------------------------------------------------------------

    def _dense_auxiliary_loss(
        self,
        physics_feat_map: Tensor,
        inst_mask: Tensor,
        phys_lut: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        """Per-pixel classification loss using instance labels broadcast to pixels."""
        B, V, C, H, W = physics_feat_map.shape
        device = physics_feat_map.device

        # Build per-pixel GT from instance_mask + phys_label_map
        # inst_mask: [B, V, H, W], phys_lut: [B, max_id+1]
        pixel_gt = torch.zeros(B, V, H, W, dtype=torch.long, device=device)
        for b in range(B):
            lut = phys_lut[b]
            mask_b = inst_mask[b]  # [V, H, W]
            clamped = mask_b.clamp(0, lut.shape[0] - 1)
            pixel_gt[b] = lut[clamped]
            # Zero out pixels whose instance_id is out of range
            pixel_gt[b][mask_b >= lut.shape[0]] = 0
            pixel_gt[b][mask_b == int(self.cfg.ignore_id)] = 0

        if valid_mask is not None:
            vm = valid_mask
            if vm.ndim == 4 and vm.shape[-1] == 1:
                vm = vm.squeeze(-1)
            pixel_gt[~vm.bool()] = 0

        # Shift to 0-indexed; mark ignore as special value
        ignore_mask = (pixel_gt == 0)
        pixel_gt_shifted = pixel_gt - 1
        pixel_gt_shifted[ignore_mask] = -100  # F.cross_entropy ignore_index

        logits = self.dense_proj(
            physics_feat_map.reshape(B * V, C, H, W)
        )  # [B*V, num_classes, H, W]

        loss = F.cross_entropy(
            logits,
            pixel_gt_shifted.reshape(B * V, H, W),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    # ------------------------------------------------------------------

    def forward(
        self,
        prediction: DecoderOutput | None,
        batch: BatchedExample,
        gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.extra_logs: dict[str, Tensor] = {}

        def _fallback_device():
            if prediction is not None and hasattr(prediction, "color"):
                return prediction.color.device
            if depth_dict is not None:
                for v in depth_dict.values():
                    if hasattr(v, "device"):
                        return v.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _zero = lambda: torch.tensor(0.0, device=_fallback_device(), dtype=torch.float32)

        if depth_dict is None:
            return _zero()

        physics_feat_map: Tensor | None = depth_dict.get("physics_feat_map")
        inst_mask: Tensor | None = depth_dict.get("instance_mask")
        phys_lut: Tensor | None = depth_dict.get("phys_label_map")
        valid_mask: Tensor | None = depth_dict.get("instance_valid_mask")
        inst_feat_map: Tensor | None = depth_dict.get("instance_feat_map") if self.cfg.use_inst_feat else None

        if physics_feat_map is None or inst_mask is None or phys_lut is None:
            if global_step % 200 == 0:
                logger.info(
                    "[LossPhys step=%d] missing data: physics_feat_map=%s inst_mask=%s phys_lut=%s",
                    global_step,
                    physics_feat_map is not None,
                    inst_mask is not None,
                    phys_lut is not None,
                )
            return _zero()

        B, V, C, H, W = physics_feat_map.shape
        device = physics_feat_map.device

        # Resolve focal alpha
        alpha_t = None
        if self.cfg.focal_alpha is not None:
            alpha_t = torch.tensor(self.cfg.focal_alpha, device=device, dtype=torch.float32)

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_correct = 0
        total_instances = 0

        for b in range(B):
            feat_b = physics_feat_map[b]  # [V, C, H, W]
            mask_b = inst_mask[b]         # [V, H, W]
            lut_b = phys_lut[b]           # [max_id+1]
            vm_b = valid_mask[b] if valid_mask is not None else None
            ifeat_b = inst_feat_map[b] if inst_feat_map is not None else None

            pooled, labels = self._pool_per_instance(feat_b, mask_b, lut_b, ifeat_b, vm_b)

            if pooled.shape[0] == 0:
                continue

            logits = self.classifier(pooled.float())
            loss_b = focal_loss(logits, labels, gamma=float(self.cfg.focal_gamma), alpha=alpha_t)
            total_loss = total_loss + loss_b

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_instances += labels.shape[0]

        if B > 0:
            total_loss = total_loss / B

        # Dense auxiliary loss
        dense_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.cfg.dense_aux_weight > 0 and hasattr(self, "dense_proj"):
            dense_loss = self._dense_auxiliary_loss(
                physics_feat_map, inst_mask, phys_lut, valid_mask,
            )
            total_loss = total_loss + float(self.cfg.dense_aux_weight) * dense_loss

        loss = float(self.cfg.weight) * total_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        acc = total_correct / max(total_instances, 1)
        self.extra_logs = {
            "phys_loss_raw": total_loss.detach(),
            "phys_acc": torch.tensor(acc, device=device),
            "phys_num_instances": torch.tensor(float(total_instances), device=device),
            "phys_dense_loss": dense_loss.detach(),
        }

        if global_step % 200 == 0:
            logger.info(
                "[LossPhys step=%d] loss=%.4f acc=%.3f instances=%d dense=%.4f",
                global_step, total_loss.item(), acc, total_instances, dense_loss.item(),
            )

        return loss

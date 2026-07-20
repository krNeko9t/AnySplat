"""Physics classification loss (formula only — no learnable params).

Layer contract:
  - Predictions: ``depth_dict["physics_prediction"]`` → PhysicsPrediction
  - Targets:     ``depth_dict["physics_target"]``     → PhysicsTarget | list[PhysicsTarget]
  - Optional dense aux also needs ``depth_dict["instance_mask"]`` [B,V,H,W]

Classifier / pooling live on the encoder. This module only resolves
(logits, labels) and applies focal CE.

Where to edit:
  - New loss formula          → this file
  - New pred/target alignment → ``resolve_instance_ce``
  - New head / classifier     → ``src/model/heads/physics/``, not here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.physics.types import PhysicsTarget
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.outputs import PhysicsPrediction
from .loss import Loss

logger = logging.getLogger(__name__)


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Multi-class focal loss. logits [N,C], targets [N]."""
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    p_t = torch.exp(-ce)
    loss = ((1 - p_t) ** gamma) * ce
    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else loss.sum()
    return loss


def resolve_instance_ce(
    pred: PhysicsPrediction,
    target: PhysicsTarget | list[PhysicsTarget],
    ignore_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Align encoder instance logits with PhysicsTarget LUTs.

    Returns logits [K, num_classes], labels [K] (0-indexed; ignore dropped).
    """
    if pred.instance_logits is None or pred.instance_ids is None:
        raise ValueError(
            "PhysicsPrediction.instance_logits/ids is None — "
            "encoder must run PhysicsClassifier with instance masks."
        )

    targets = target if isinstance(target, list) else [target]
    if len(targets) != len(pred.instance_logits):
        raise ValueError(
            f"batch size mismatch: {len(targets)} targets vs "
            f"{len(pred.instance_logits)} predictions"
        )

    logit_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    for logits_b, ids_b, tgt_b in zip(
        pred.instance_logits, pred.instance_ids, targets, strict=True
    ):
        if logits_b.numel() == 0:
            continue
        lut = tgt_b.label_lut.to(device=ids_b.device)
        in_range = ids_b < lut.shape[0]
        ids_ok = ids_b[in_range]
        logits_ok = logits_b[in_range]
        phys = lut[ids_ok]
        keep = phys != ignore_id
        if not keep.any():
            continue
        label_chunks.append(phys[keep] - 1)  # 1-indexed → 0-indexed CE
        logit_chunks.append(logits_ok[keep])

    if not logit_chunks:
        device = pred.instance_logits[0].device
        n_cls = pred.instance_logits[0].shape[-1]
        return (
            torch.empty((0, n_cls), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    return torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)


def _as_target_list(
    target: PhysicsTarget | list[PhysicsTarget],
    batch_size: int,
) -> list[PhysicsTarget]:
    if isinstance(target, list):
        return target
    if batch_size != 1:
        raise ValueError(
            "PhysicsTarget is a single object but batch_size>1; "
            "pass list[PhysicsTarget] (one per sample)."
        )
    return [target]


@dataclass
class LossPhysCfg:
    weight: float = 1.0
    num_classes: int = 4
    ignore_id: int = 0
    focal_gamma: float = 2.0
    focal_alpha: Optional[list[float]] = None
    dense_aux_weight: float = 0.0


@dataclass
class LossPhysCfgWrapper:
    phys: LossPhysCfg


class LossPhys(Loss[LossPhysCfg, LossPhysCfgWrapper]):
    """Focal CE over PhysicsPrediction instance logits. No learnable params."""

    def forward(
        self,
        prediction: DecoderOutput | None,
        batch: BatchedExample,
        gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.extra_logs: dict[str, Tensor] = {}

        def _device():
            if depth_dict is not None:
                pred = depth_dict.get("physics_prediction")
                if isinstance(pred, PhysicsPrediction):
                    if pred.feat_map is not None:
                        return pred.feat_map.device
                    if pred.instance_logits:
                        return pred.instance_logits[0].device
                tgt = depth_dict.get("physics_target")
                if isinstance(tgt, PhysicsTarget):
                    return tgt.label_lut.device
                if isinstance(tgt, list) and tgt:
                    return tgt[0].label_lut.device
                for v in depth_dict.values():
                    if hasattr(v, "device"):
                        return v.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _zero = lambda: torch.tensor(0.0, device=_device(), dtype=torch.float32)

        if depth_dict is None:
            return _zero()

        pred: PhysicsPrediction | None = depth_dict.get("physics_prediction")
        target = depth_dict.get("physics_target")

        if pred is None or target is None:
            if global_step % 200 == 0:
                logger.info(
                    "[LossPhys step=%d] missing physics_prediction=%s physics_target=%s",
                    global_step,
                    pred is not None,
                    target is not None,
                )
            return _zero()

        logits, labels = resolve_instance_ce(
            pred, target, ignore_id=int(self.cfg.ignore_id)
        )
        device = logits.device

        if logits.shape[0] == 0:
            self.extra_logs = {
                "phys_loss_raw": torch.tensor(0.0, device=device),
                "phys_acc": torch.tensor(0.0, device=device),
                "phys_num_instances": torch.tensor(0.0, device=device),
                "phys_dense_loss": torch.tensor(0.0, device=device),
            }
            return _zero()

        alpha_t = None
        if self.cfg.focal_alpha is not None:
            alpha_t = torch.tensor(
                self.cfg.focal_alpha, device=device, dtype=torch.float32
            )

        raw = focal_loss(
            logits, labels, gamma=float(self.cfg.focal_gamma), alpha=alpha_t
        )

        dense_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if float(self.cfg.dense_aux_weight) > 0 and pred.dense_logits is not None:
            inst_mask = depth_dict.get("instance_mask")
            if inst_mask is not None:
                dense_loss = self._dense_aux(pred, target, inst_mask)

        total = raw + float(self.cfg.dense_aux_weight) * dense_loss
        loss = float(self.cfg.weight) * total
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.extra_logs = {
            "phys_loss_raw": raw.detach(),
            "phys_acc": acc,
            "phys_num_instances": torch.tensor(float(labels.shape[0]), device=device),
            "phys_dense_loss": dense_loss.detach(),
        }

        if global_step % 200 == 0:
            logger.info(
                "[LossPhys step=%d] loss=%.4f acc=%.3f instances=%d dense=%.4f",
                global_step,
                raw.item(),
                acc.item(),
                labels.shape[0],
                dense_loss.item(),
            )

        return loss

    def _dense_aux(
        self,
        pred: PhysicsPrediction,
        target: PhysicsTarget | list[PhysicsTarget],
        inst_mask: Tensor,
    ) -> Tensor:
        """Per-pixel CE: broadcast PhysicsTarget LUT through instance_mask."""
        assert pred.dense_logits is not None
        B, V, n_cls, H, W = pred.dense_logits.shape
        targets = _as_target_list(target, B)
        device = pred.dense_logits.device

        pixel_gt = torch.zeros(B, V, H, W, dtype=torch.long, device=device)
        ignore = int(self.cfg.ignore_id)
        for b in range(B):
            lut = targets[b].label_lut.to(device)
            mask_b = inst_mask[b]
            clamped = mask_b.clamp(0, lut.shape[0] - 1)
            pixel_gt[b] = lut[clamped]
            pixel_gt[b][mask_b >= lut.shape[0]] = ignore
            pixel_gt[b][mask_b == ignore] = ignore

        ignore_mask = pixel_gt == ignore
        pixel_gt_shifted = pixel_gt - 1
        pixel_gt_shifted[ignore_mask] = -100

        return F.cross_entropy(
            pred.dense_logits.reshape(B * V, n_cls, H, W),
            pixel_gt_shifted.reshape(B * V, H, W),
            ignore_index=-100,
            reduction="mean",
        )

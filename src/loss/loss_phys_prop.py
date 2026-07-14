"""Physics property regression loss (formula only — no learnable params).

Layer contract:
  - Predictions: ``depth_dict["physics_property_prediction"]`` → PhysicsPropertyPrediction
  - Targets:     ``depth_dict["physics_property_target"]``     → PhysicsPropertyTarget | list

Readout / pooling live on the encoder. This module only resolves
(pred, gt) and applies MSE on mean + weighted MSE on log_var.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.physics.types import PhysicsPropertyTarget
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.encoder.physics_property_prediction import PhysicsPropertyPrediction
from .loss import Loss

logger = logging.getLogger(__name__)


def resolve_instance_prop(
    pred: PhysicsPropertyPrediction,
    target: PhysicsPropertyTarget | list[PhysicsPropertyTarget],
) -> tuple[Tensor, Tensor]:
    """Align encoder instance values with PhysicsPropertyTarget LUTs.

    Returns pred_vals [K, P, 2], gt_vals [K, P, 2] (valid instances only).
    """
    if pred.instance_values is None or pred.instance_ids is None:
        raise ValueError(
            "PhysicsPropertyPrediction.instance_values/ids is None — "
            "encoder must run PhysicsPropertyReadout with instance masks."
        )

    targets = target if isinstance(target, list) else [target]
    if len(targets) != len(pred.instance_values):
        raise ValueError(
            f"batch size mismatch: {len(targets)} targets vs "
            f"{len(pred.instance_values)} predictions"
        )

    pred_chunks: list[Tensor] = []
    gt_chunks: list[Tensor] = []

    for vals_b, ids_b, tgt_b in zip(
        pred.instance_values, pred.instance_ids, targets, strict=True
    ):
        if vals_b.numel() == 0:
            continue
        mean_lut = tgt_b.mean_lut.to(device=ids_b.device)
        log_var_lut = tgt_b.log_var_lut.to(device=ids_b.device)
        valid = tgt_b.valid.to(device=ids_b.device)

        in_range = ids_b < valid.shape[0]
        ids_ok = ids_b[in_range]
        vals_ok = vals_b[in_range]
        keep = valid[ids_ok]
        if not keep.any():
            continue
        ids_keep = ids_ok[keep]
        pred_chunks.append(vals_ok[keep])
        gt_mean = mean_lut[ids_keep]
        gt_log_var = log_var_lut[ids_keep]
        gt_chunks.append(torch.stack([gt_mean, gt_log_var], dim=-1))

    if not pred_chunks:
        device = pred.instance_values[0].device
        p = pred.instance_values[0].shape[-2] if pred.instance_values[0].ndim >= 2 else 0
        if p == 0 and pred.property_names is not None:
            p = len(pred.property_names)
        return (
            torch.empty((0, p, 2), device=device, dtype=torch.float32),
            torch.empty((0, p, 2), device=device, dtype=torch.float32),
        )

    return torch.cat(pred_chunks, dim=0), torch.cat(gt_chunks, dim=0)


@dataclass
class LossPhysPropCfg:
    weight: float = 1.0
    var_weight: float = 1.0


@dataclass
class LossPhysPropCfgWrapper:
    phys_prop: LossPhysPropCfg


class LossPhysProp(Loss[LossPhysPropCfg, LossPhysPropCfgWrapper]):
    """MSE on property mean + weighted MSE on log_var. No learnable params."""

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
                pred = depth_dict.get("physics_property_prediction")
                if isinstance(pred, PhysicsPropertyPrediction):
                    if pred.feat_map is not None:
                        return pred.feat_map.device
                    if pred.instance_values:
                        return pred.instance_values[0].device
                tgt = depth_dict.get("physics_property_target")
                if isinstance(tgt, PhysicsPropertyTarget):
                    return tgt.mean_lut.device
                if isinstance(tgt, list) and tgt:
                    return tgt[0].mean_lut.device
                for v in depth_dict.values():
                    if hasattr(v, "device"):
                        return v.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _zero = lambda: torch.tensor(0.0, device=_device(), dtype=torch.float32)

        if depth_dict is None:
            return _zero()

        pred: PhysicsPropertyPrediction | None = depth_dict.get(
            "physics_property_prediction"
        )
        target = depth_dict.get("physics_property_target")

        if pred is None or target is None:
            if global_step % 200 == 0:
                logger.info(
                    "[LossPhysProp step=%d] missing pred=%s target=%s",
                    global_step,
                    pred is not None,
                    target is not None,
                )
            return _zero()

        pred_vals, gt_vals = resolve_instance_prop(pred, target)
        device = pred_vals.device

        if pred_vals.shape[0] == 0:
            self.extra_logs = {
                "phys_prop_loss_raw": torch.tensor(0.0, device=device),
                "phys_prop_mean_mse": torch.tensor(0.0, device=device),
                "phys_prop_log_var_mse": torch.tensor(0.0, device=device),
                "phys_prop_num_instances": torch.tensor(0.0, device=device),
            }
            return _zero()

        mean_mse = F.mse_loss(pred_vals[..., 0], gt_vals[..., 0])
        log_var_mse = F.mse_loss(pred_vals[..., 1], gt_vals[..., 1])
        raw = mean_mse + float(self.cfg.var_weight) * log_var_mse
        loss = float(self.cfg.weight) * raw
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.extra_logs = {
            "phys_prop_loss_raw": raw.detach(),
            "phys_prop_mean_mse": mean_mse.detach(),
            "phys_prop_log_var_mse": log_var_mse.detach(),
            "phys_prop_num_instances": torch.tensor(
                float(pred_vals.shape[0]), device=device
            ),
        }

        if global_step % 200 == 0:
            logger.info(
                "[LossPhysProp step=%d] loss=%.4f mean_mse=%.4f log_var_mse=%.4f n=%d",
                global_step,
                raw.item(),
                mean_mse.item(),
                log_var_mse.item(),
                pred_vals.shape[0],
            )

        return loss

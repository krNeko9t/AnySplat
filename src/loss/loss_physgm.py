"""PhysGM-style property regression loss (formula only — no learnable params).

Faithful copy of PhysGM's E/nu supervision (model/physgm.py):
  robust Gaussian NLL (var floor 1e-2, per-element clamp at 100) + MSE on mu,
  applied per property on z-scored targets (see PHYSGM_NORMALIZATION).

Layer contract:
  - Predictions: ``depth_dict["physgm_prediction"]`` → PhysGMPrediction
  - Targets:     ``depth_dict["physgm_target"]``     → PhysGMTarget | list

Readout / pooling live on the encoder. This module only resolves
(pred, gt) and applies the formula.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.physics.types import PhysGMTarget
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.encoder.predictions import PhysGMPrediction
from .loss import Loss

logger = logging.getLogger(__name__)


def resolve_instance_physgm(
    pred: PhysGMPrediction,
    target: PhysGMTarget | list[PhysGMTarget],
) -> tuple[Tensor, Tensor, Tensor]:
    """Align encoder instance predictions with PhysGMTarget LUTs.

    Returns (mu [K, P], var [K, P], gt [K, P]) over valid instances only.
    """
    if pred.instance_mu is None or pred.instance_var is None or pred.instance_ids is None:
        raise ValueError(
            "PhysGMPrediction.instance_mu/var/ids is None — "
            "encoder must run PhysGMReadout with instance masks."
        )

    targets = target if isinstance(target, list) else [target]
    if len(targets) != len(pred.instance_mu):
        raise ValueError(
            f"batch size mismatch: {len(targets)} targets vs "
            f"{len(pred.instance_mu)} predictions"
        )

    mu_chunks: list[Tensor] = []
    var_chunks: list[Tensor] = []
    gt_chunks: list[Tensor] = []

    for mu_b, var_b, ids_b, tgt_b in zip(
        pred.instance_mu, pred.instance_var, pred.instance_ids, targets, strict=True
    ):
        if mu_b.numel() == 0:
            continue
        value_lut = tgt_b.value_lut.to(device=ids_b.device)
        valid = tgt_b.valid.to(device=ids_b.device)

        in_range = ids_b < valid.shape[0]
        ids_ok = ids_b[in_range]
        keep = valid[ids_ok]
        if not keep.any():
            continue
        ids_keep = ids_ok[keep]
        mu_chunks.append(mu_b[in_range][keep])
        var_chunks.append(var_b[in_range][keep])
        gt_chunks.append(value_lut[ids_keep])

    if not mu_chunks:
        device = pred.instance_mu[0].device
        p = len(pred.property_names) if pred.property_names is not None else 0
        empty = torch.empty((0, p), device=device, dtype=torch.float32)
        return empty, empty.clone(), empty.clone()

    return (
        torch.cat(mu_chunks, dim=0),
        torch.cat(var_chunks, dim=0),
        torch.cat(gt_chunks, dim=0),
    )


def robust_gaussian_nll(mu: Tensor, var: Tensor, target: Tensor) -> Tensor:
    """PhysGM's robust_gaussian_nll_loss: var floor 1e-2, per-element clamp 100."""
    mu = mu.float()
    var = var.float().clamp(min=1e-2)
    target = target.float()
    nll = 0.5 * math.log(2 * math.pi) + 0.5 * torch.log(var) + 0.5 * (target - mu) ** 2 / var
    return nll.clamp(max=100.0).mean()


@dataclass
class LossPhysGMCfg:
    weight: float = 1.0
    # PhysGM: E_total = NLL + mse_weight * MSE per property (mse_weight = 1.0).
    mse_weight: float = 1.0


@dataclass
class LossPhysGMCfgWrapper:
    physgm: LossPhysGMCfg


class LossPhysGM(Loss[LossPhysGMCfg, LossPhysGMCfgWrapper]):
    """Per-property Gaussian NLL + MSE (PhysGM formula). No learnable params."""

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
                pred = depth_dict.get("physgm_prediction")
                if isinstance(pred, PhysGMPrediction) and pred.instance_mu:
                    return pred.instance_mu[0].device
                for v in depth_dict.values():
                    if hasattr(v, "device"):
                        return v.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _zero(pred: PhysGMPrediction | None = None) -> Tensor:
            """Scalar 0 that stays on the autograd graph when predictions exist.

            Bare ``torch.tensor(0.)`` has no grad_fn; with physgm as the sole
            loss, an empty-supervision batch would then crash on backward.
            """
            terms: list[Tensor] = []
            if pred is not None:
                if pred.instance_mu:
                    terms.extend(m.float().sum() for m in pred.instance_mu)
                if pred.instance_var:
                    terms.extend(v.float().sum() for v in pred.instance_var)
            if terms:
                acc = terms[0]
                for t in terms[1:]:
                    acc = acc + t
                return acc * 0.0
            return torch.zeros((), device=_device(), dtype=torch.float32, requires_grad=True)

        if depth_dict is None:
            return _zero()

        pred: PhysGMPrediction | None = depth_dict.get("physgm_prediction")
        target = depth_dict.get("physgm_target")

        if pred is None or target is None:
            if global_step % 200 == 0:
                logger.info(
                    "[LossPhysGM step=%d] missing pred=%s target=%s",
                    global_step,
                    pred is not None,
                    target is not None,
                )
            return _zero(pred)

        mu, var, gt = resolve_instance_physgm(pred, target)
        device = mu.device

        if mu.shape[0] == 0:
            self.extra_logs = {
                "physgm_loss_raw": torch.tensor(0.0, device=device),
                "physgm_num_instances": torch.tensor(0.0, device=device),
            }
            return _zero(pred)

        prop_names = pred.property_names or tuple(
            f"prop_{i}" for i in range(mu.shape[1])
        )
        raw = mu.new_zeros(())
        for p_i, name in enumerate(prop_names):
            nll = robust_gaussian_nll(mu[:, p_i], var[:, p_i], gt[:, p_i])
            mse = F.mse_loss(mu[:, p_i].float(), gt[:, p_i].float())
            raw = raw + nll + float(self.cfg.mse_weight) * mse
            self.extra_logs[f"physgm_{name}_nll"] = nll.detach()
            self.extra_logs[f"physgm_{name}_mse"] = mse.detach()

        loss = float(self.cfg.weight) * raw
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.extra_logs["physgm_loss_raw"] = raw.detach()
        self.extra_logs["physgm_num_instances"] = torch.tensor(
            float(mu.shape[0]), device=device
        )

        if global_step % 200 == 0:
            logger.info(
                "[LossPhysGM step=%d] loss=%.4f n=%d %s",
                global_step,
                raw.item(),
                mu.shape[0],
                " ".join(
                    f"{name}: nll={self.extra_logs[f'physgm_{name}_nll'].item():.4f} "
                    f"mse={self.extra_logs[f'physgm_{name}_mse'].item():.4f}"
                    for name in prop_names
                ),
            )

        return loss

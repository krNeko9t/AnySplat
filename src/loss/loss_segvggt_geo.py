"""SegVGGT geometry loss: L_geo = lambda_camera * L_camera + lambda_depth * L_depth.

The paper (Sec. 3.3 / 4.1) states the geometry losses "directly follow VGGT" and that
a frozen pretrained VGGT is used as a *teacher* to distil robust depth / camera targets
(mitigating overfit to noisy GT).  This module implements the loss *forms*; the
supervision *source* is decoupled and provided by :class:`SegVGGTWrapper` through
``depth_dict['segvggt_geo_target']`` -- either clean manifest GT (``geo_supervision:
gt``, always available) or a frozen VGGT teacher (``geo_supervision: teacher``,
paper-faithful).  Both feed the same target contract, so this loss is source-agnostic.

Camera term: Huber on the 9-D pose encoding, summed over the camera head's refinement
iterations with a ``gamma``-decay (VGGT / IGGT convention, see :mod:`loss_huber`).

Depth term: VGGT depth is only up-to-scale, so predictions are aligned to the target
with a single per-sequence median scale (matching the paper's eval protocol) before an
L1 + first-order gradient-matching penalty on valid pixels.

Target contract (``depth_dict['segvggt_geo_target']``), all for the S encoder frames:
  - pose_enc : Float [B, S, 9]   target 9-D pose encoding (absT_quaR_FoV)
  - depth    : Float [B, S, H, W] target depth
  - valid    : Bool  [B, S, H, W] valid-depth mask
Plus the predictions (``depth_dict``):
  - pred_pose_enc_list : list of Float [B, S, 9]
  - depth              : Float [B, S, H, W, 1] or [B, S, H, W]
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
from .loss_huber import huber_loss

logger = logging.getLogger(__name__)


@dataclass
class LossSegVGGTGeoCfg:
    weight: float = 1.0
    lambda_camera: float = 5.0   # paper A.4
    lambda_depth: float = 1.0
    camera_gamma: float = 0.6    # decay across camera-head iterations
    weight_T: float = 1.0
    weight_R: float = 1.0
    weight_fl: float = 0.5
    depth_grad_weight: float = 0.5  # first-order gradient-matching term
    align_scale: bool = True        # per-sequence median scale alignment (up-to-scale)


@dataclass
class LossSegVGGTGeoCfgWrapper:
    segvggt_geo: LossSegVGGTGeoCfg


class LossSegVGGTGeo(Loss[LossSegVGGTGeoCfg, LossSegVGGTGeoCfgWrapper]):
    """Camera (Huber pose) + depth (scale-aligned L1 + gradient) geometry loss."""

    def __init__(self, cfg: LossSegVGGTGeoCfgWrapper) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------ #
    def _camera_loss(self, pred_pose_enc_list: list[Tensor], gt_pose_enc: Tensor) -> Tensor:
        n = len(pred_pose_enc_list)
        loss_T = loss_R = loss_fl = pred_pose_enc_list[0].new_tensor(0.0)
        for i, cur in enumerate(pred_pose_enc_list):
            w = self.cfg.camera_gamma ** (n - i - 1)
            cur = cur.float()
            lT = huber_loss(cur[..., :3], gt_pose_enc[..., :3])
            lR = huber_loss(cur[..., 3:7], gt_pose_enc[..., 3:7])
            lfl = huber_loss(cur[..., 7:], gt_pose_enc[..., 7:])
            lT = lT.clamp(-100, 100).mean()
            lR = lR.clamp(-100, 100).mean()
            lfl = lfl.clamp(-100, 100).mean()
            loss_T = loss_T + w * lT
            loss_R = loss_R + w * lR
            loss_fl = loss_fl + w * lfl
        loss_T, loss_R, loss_fl = loss_T / n, loss_R / n, loss_fl / n
        return (
            self.cfg.weight_T * loss_T
            + self.cfg.weight_R * loss_R
            + self.cfg.weight_fl * loss_fl
        )

    def _depth_loss(self, pred: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
        """pred/gt/valid: [B, S, H, W].  Scale-aligned L1 + first-order gradient."""
        if valid.sum() == 0:
            return pred.new_tensor(0.0)
        pred = pred.float()
        gt = gt.float()

        if self.cfg.align_scale:
            # single per-sequence median scale (predictions are up-to-scale)
            b, s = pred.shape[:2]
            pred_flat = pred.reshape(b, -1)
            gt_flat = gt.reshape(b, -1)
            v_flat = valid.reshape(b, -1)
            scale = torch.ones(b, device=pred.device)
            for i in range(b):
                m = v_flat[i] & (pred_flat[i] > 1e-6)
                if m.sum() > 0:
                    scale[i] = (
                        gt_flat[i][m].median() / pred_flat[i][m].median().clamp_min(1e-6)
                    )
            pred = pred * scale.view(b, 1, 1, 1)

        l1 = huber_loss(pred[valid], gt[valid]).mean()

        # first-order gradient matching (edge-aware depth smoothness vs GT)
        grad = pred.new_tensor(0.0)
        if self.cfg.depth_grad_weight > 0:
            dx_p = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
            dx_g = (gt[..., :, 1:] - gt[..., :, :-1]).abs()
            dy_p = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
            dy_g = (gt[..., 1:, :] - gt[..., :-1, :]).abs()
            vx = valid[..., :, 1:] & valid[..., :, :-1]
            vy = valid[..., 1:, :] & valid[..., :-1, :]
            gx = (dx_p - dx_g).abs()[vx]
            gy = (dy_p - dy_g).abs()[vy]
            parts = [t.mean() for t in (gx, gy) if t.numel() > 0]
            if parts:
                grad = torch.stack(parts).mean()
        return l1 + self.cfg.depth_grad_weight * grad

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def metrics(self, depth_dict: dict | None) -> dict[str, Tensor]:
        """Read-only geometry-drift numbers.  Never enters the loss.

        Reuses exactly the terms :meth:`forward` computes, so ``geo_camera`` /
        ``geo_depth`` stay comparable with ``loss_segvggt_camera`` /
        ``loss_segvggt_depth`` even while ``weight: 0`` keeps geometry out of the
        objective.  Adds a per-component breakdown of the *final* camera-head
        iteration -- with no geometry loss watching, "did it drift" is only useful
        if it also says *what* drifted (translation, rotation, or focal).

        Returns ``{}`` when the pieces aren't there (no target, no predictions).
        A depth target that is constant everywhere is the manifest's
        depth-less placeholder (see ``DatasetManifest``: depth of ones, valid
        all-True), so the depth terms are dropped rather than reported as 0.
        """
        out: dict[str, Tensor] = {}
        if depth_dict is None:
            return out
        target = depth_dict.get("segvggt_geo_target")
        if target is None:
            return out

        pose_list = depth_dict.get("pred_pose_enc_list")
        gt_pose = target.get("pose_enc")
        if pose_list and gt_pose is not None:
            gt_pose = gt_pose.float()
            out["geo_camera"] = self._camera_loss(pose_list, gt_pose)
            last = pose_list[-1].float()
            out["geo_camera_T_last"] = huber_loss(last[..., :3], gt_pose[..., :3]).clamp(-100, 100).mean()
            out["geo_camera_R_last"] = huber_loss(last[..., 3:7], gt_pose[..., 3:7]).clamp(-100, 100).mean()
            out["geo_camera_fl_last"] = huber_loss(last[..., 7:], gt_pose[..., 7:]).clamp(-100, 100).mean()

        pred_depth = depth_dict.get("depth")
        gt_depth, valid = target.get("depth"), target.get("valid")
        if pred_depth is not None and gt_depth is not None and valid is not None:
            if pred_depth.dim() == 5:
                pred_depth = pred_depth.squeeze(-1)
            if valid.dim() == 5:
                valid = valid.squeeze(-1)
            gt_depth = gt_depth.float()
            valid = valid.bool()
            if gt_depth.numel() > 0 and (gt_depth.amax() - gt_depth.amin()) > 1e-6:
                out["geo_depth"] = self._depth_loss(pred_depth, gt_depth, valid)
                # ...and the same number divided by the median GT depth. The raw
                # term carries the dataset's depth unit (the manifest hands over
                # ScanNet++ depth in *millimetres*, median ~2263), so 106 reads as
                # alarming when it is 4.7% relative error. The ratio is the one
                # that can be compared across arms, datasets and resolutions.
                if valid.any():
                    med = gt_depth[valid].median().clamp_min(1e-6)
                    out["geo_depth_rel"] = out["geo_depth"] / med
        return {k: v.detach() for k, v in out.items()}

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
        if depth_dict is None:
            return torch.tensor(0.0, device=device)
        target = depth_dict.get("segvggt_geo_target")
        if target is None:
            return torch.tensor(0.0, device=device)

        l_cam = None
        pose_list = depth_dict.get("pred_pose_enc_list")
        if pose_list is not None and target.get("pose_enc") is not None:
            l_cam = self._camera_loss(pose_list, target["pose_enc"].float())

        l_depth = None
        pred_depth = depth_dict.get("depth")
        if (
            pred_depth is not None
            and target.get("depth") is not None
            and target.get("valid") is not None
        ):
            if pred_depth.dim() == 5:  # [B, S, H, W, 1]
                pred_depth = pred_depth.squeeze(-1)
            valid = target["valid"]
            if valid.dim() == 5:
                valid = valid.squeeze(-1)
            l_depth = self._depth_loss(pred_depth, target["depth"], valid.bool())

        device = (l_cam if l_cam is not None else l_depth).device if (
            l_cam is not None or l_depth is not None
        ) else device
        zero = torch.tensor(0.0, device=device)
        l_cam = zero if l_cam is None else l_cam
        l_depth = zero if l_depth is None else l_depth

        loss = self.cfg.weight * (
            self.cfg.lambda_camera * l_cam + self.cfg.lambda_depth * l_depth
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"[LossSegVGGTGeo step={global_step}] non-finite loss={loss.item()}: "
                f"camera={float(l_cam.detach())} depth={float(l_depth.detach())}"
            )

        self.extra_logs = {
            "loss_segvggt_camera": l_cam.detach(),
            "loss_segvggt_depth": l_depth.detach(),
        }
        if global_step % 100 == 0:
            logger.info(
                "[LossSegVGGTGeo step=%d] camera=%.4f depth=%.4f",
                global_step, float(l_cam.detach()), float(l_depth.detach()),
            )
        return loss

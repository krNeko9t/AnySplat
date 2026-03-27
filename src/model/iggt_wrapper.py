"""IGGTWrapper -- training / validation / test for IGGT (encoder only, instance features)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..evaluation.metrics import abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_huber import HuberLoss
from ..misc.image_io import prep_image
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from .base_wrapper import (
    BaseModelWrapper,
    OptimizerCfg,
    TestCfg,
    TrainCfg,
    visualize_instance_features,
)

logger = logging.getLogger(__name__)


class IGGTWrapper(BaseModelWrapper):
    """Training loop for IGGT (encoder only, instance segmentation)."""

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(optimizer_cfg, test_cfg, train_cfg, model, losses, step_tracker)

        if getattr(self.model.encoder, "pred_pose", False):
            self.loss_pose = HuberLoss(
                alpha=self.train_cfg.pose_loss_alpha,
                delta=self.train_cfg.pose_loss_delta,
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch = self.combine_batches(batch)
        batch: BatchedExample = self.data_shim(batch)
        b, v_ctx, c, h, w = batch["context"]["image"].shape

        context_image = (batch["context"]["image"] + 1) / 2
        if "target" in batch and "image" in batch["target"]:
            target_image = (batch["target"]["image"] + 1) / 2
            input_image = torch.cat([context_image, target_image], dim=1)
        else:
            input_image = context_image

        encoder_output, _ = self.model(input_image, self.global_step)
        depth_dict = encoder_output.depth_dict or {}
        infos = encoder_output.infos or {}

        self.log("train/scene_scale", infos.get("scene_scale", 0.0))

        depth_dict_for_loss = dict(depth_dict)

        if encoder_output.instance_feat_map is not None:
            depth_dict_for_loss["instance_feat_map"] = encoder_output.instance_feat_map
        if encoder_output.physics_feat_map is not None:
            depth_dict_for_loss["physics_feat_map"] = encoder_output.physics_feat_map
        if "context" in batch and "instance_mask" in batch["context"] and "target" in batch and "instance_mask" in batch["target"]:
            depth_dict_for_loss["instance_mask"] = torch.cat(
                [batch["context"]["instance_mask"], batch["target"]["instance_mask"]], dim=1,
            )
        elif "context" in batch and "instance_mask" in batch["context"]:
            depth_dict_for_loss["instance_mask"] = batch["context"]["instance_mask"]
        if "context" in batch and "valid_mask" in batch["context"] and "target" in batch and "valid_mask" in batch["target"]:
            depth_dict_for_loss["instance_valid_mask"] = torch.cat(
                [batch["context"]["valid_mask"], batch["target"]["valid_mask"]], dim=1,
            )
        elif "context" in batch and "valid_mask" in batch["context"]:
            depth_dict_for_loss["instance_valid_mask"] = batch["context"]["valid_mask"]
        # Physics label map (scene-level, shared across context/target)
        if "context" in batch and "phys_label_map" in batch["context"]:
            depth_dict_for_loss["phys_label_map"] = batch["context"]["phys_label_map"]

        with torch.amp.autocast("cuda", enabled=False):
            total_loss, loss_values = self.compute_and_log_losses(
                self.losses, None, batch, None, depth_dict_for_loss, self.global_step,
            )

        return self.finalize_training_step(total_loss, loss_values, batch)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        encoder_output, _ = self.model(
            (batch["context"]["image"] + 1) / 2,
            self.global_step,
        )
        depth_dict = encoder_output.depth_dict or {}

        if "depth" in depth_dict:
            depth_pred_raw = depth_dict["depth"]
            if depth_pred_raw is not None and depth_pred_raw.numel() > 0:
                self.log(
                    "val/depth_mean",
                    depth_pred_raw.mean(),
                )

        if self.trainer.global_rank == 0:
            logger.info(
                "validation step %s; scene = %s; context = %s",
                self.global_step, batch["scene"], batch["context"]["index"].tolist(),
            )
            if batch_idx % 100 == 0:
                context_img = inverse_normalize(batch["context"]["image"][0])

                cols = [add_label(vcat(*context_img), "Context")]

                if "depth" in depth_dict and depth_dict["depth"] is not None:
                    d = depth_dict["depth"].squeeze(-1)[0]
                    cols.append(add_label(vcat(*vis_depth_map(d)), "Depth (VGGT)"))

                comparison = hcat(*cols)
                comparison = torch.nn.functional.interpolate(
                    comparison.unsqueeze(0), scale_factor=0.5,
                    mode="bicubic", align_corners=False,
                ).squeeze(0)
                self.logger.log_image(
                    "comparison",
                    [prep_image(add_border(comparison))],
                    step=self.global_step,
                    caption=batch["scene"],
                )

                visualize_instance_features(
                    encoder_output, batch, context_img, depth_dict,
                    self.logger, self.global_step,
                )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if self.global_rank == 0 and batch_idx % 100 == 0:
            logger.info("Test step %s.", f"{batch_idx:0>6}")

        with self.benchmarker.time("encoder"):
            encoder_output, _ = self.model(
                (batch["context"]["image"] + 1) / 2, self.global_step,
            )

        if self.test_cfg.save_image and encoder_output.instance_feat_map is not None:
            (scene,) = batch["scene"]
            name = get_cfg()["wandb"]["name"]
            path = self.test_cfg.output_path / name / scene
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                encoder_output.instance_feat_map.cpu(),
                path / "instance_feat_map.pt",
            )

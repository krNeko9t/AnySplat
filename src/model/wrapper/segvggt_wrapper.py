"""SegVGGTWrapper -- encoder-only training / validation for SegVGGT.

Mirrors :class:`IGGTWrapper` (no decoder / rendering).  It runs the SegVGGT encoder
once over the concatenated context+target views and routes supervision to the losses
through ``depth_dict`` (the repo's established encoder-only convention):

  instance / FADA  (``LossSegVGGT``):
    encoder.segvggt_prediction  -> depth_dict['segvggt_prediction']
    batch instance_mask/valid   -> depth_dict['instance_mask' / 'instance_valid_mask']

  geometry         (``LossSegVGGTGeo``):
    encoder.pred_pose_enc_list  -> depth_dict['pred_pose_enc_list']
    encoder depth               -> depth_dict['depth']
    geometry target             -> depth_dict['segvggt_geo_target']  (GT or teacher)

Geometry supervision source (``segvggt_geo_supervision`` on TrainCfg, default ``gt``):
  * ``gt``      -- clean manifest camera (c2w -> w2c pose encoding) + depth. Always
                   available, needs no extra weights; the runnable default.
  * ``teacher`` -- paper-faithful: a frozen pretrained VGGT distils depth + camera.
                   Built lazily from the vendored ``src/model/vggt`` box; requires the
                   VGGT-1B weights + full env, so it is opt-in.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from src.dataset.types import BatchedExample
from src.loss import Loss
from src.loss.loss_huber import extri_intri_to_pose_encoding
from src.misc.step_tracker import StepTracker
from src.misc.utils import inverse_normalize, vis_depth_map
from src.visualization.annotation import add_label
from src.visualization.layout import add_border, hcat, vcat
from src.misc.image_io import prep_image
from .base_wrapper import BaseModelWrapper, OptimizerCfg, TestCfg, TrainCfg

logger = logging.getLogger(__name__)


def _cat_ctx_tgt(batch: BatchedExample, key: str):
    """Concatenate context/target view tensors along the view dim when both exist."""
    ctx = batch.get("context", {})
    tgt = batch.get("target", {})
    if key in ctx and key in tgt:
        return torch.cat([ctx[key], tgt[key]], dim=1)
    if key in ctx:
        return ctx[key]
    return None


class SegVGGTWrapper(BaseModelWrapper):
    """Training loop for SegVGGT (encoder only, end-to-end instance segmentation)."""

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
        self.geo_supervision = str(
            getattr(train_cfg, "segvggt_geo_supervision", "gt")
        ).lower()
        self._teacher: nn.Module | None = None

    # ------------------------------------------------------------------ #
    # geometry target
    # ------------------------------------------------------------------ #
    def _geo_target_from_gt(self, batch: BatchedExample) -> dict | None:
        """Build the geometry target from clean manifest GT (camera + depth)."""
        extr = _cat_ctx_tgt(batch, "extrinsics")   # [B, S, 4, 4] c2w
        intr = _cat_ctx_tgt(batch, "intrinsics")   # [B, S, 3, 3] normalised
        if extr is None or intr is None:
            return None
        w2c = torch.linalg.inv(extr.float())        # camera-from-world
        image_hw = batch["context"]["image"].shape[-2:]
        pose_enc = extri_intri_to_pose_encoding(w2c[:, :, :3, :4], intr.float(), image_hw)

        depth = _cat_ctx_tgt(batch, "depth")        # [B, S, H, W] or [B,S,H,W,1]
        valid = _cat_ctx_tgt(batch, "valid_mask")
        if depth is not None and depth.dim() == 5:
            depth = depth.squeeze(-1)
        return {"pose_enc": pose_enc, "depth": depth, "valid": valid}

    @torch.no_grad()
    def _geo_target_from_teacher(self, input_image: torch.Tensor) -> dict | None:
        """Paper-faithful geometry target: frozen pretrained VGGT teacher (opt-in)."""
        if self._teacher is None:
            try:
                from src.model.vggt.models.vggt import VGGT

                self._teacher = VGGT.from_pretrained("facebook/VGGT-1B").eval()
                self._teacher.to(input_image.device)
                for p in self._teacher.parameters():
                    p.requires_grad_(False)
            except Exception as e:  # keep training runnable if teacher can't load
                logger.warning("VGGT teacher unavailable (%s); geometry loss -> 0", e)
                self._teacher = None
                return None
        with torch.autocast(device_type=input_image.device.type, dtype=torch.bfloat16,
                            enabled=input_image.device.type == "cuda"):
            preds = self._teacher(input_image)
        pose_enc = preds.get("pose_enc")
        depth = preds.get("depth")
        conf = preds.get("depth_conf")
        if depth is not None and depth.dim() == 5:
            depth = depth.squeeze(-1)
        valid = None if conf is None else (conf > conf.new_tensor(0.0))
        return {"pose_enc": pose_enc, "depth": depth, "valid": valid}

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        batch = self.combine_batches(batch)
        batch: BatchedExample = self.data_shim(batch)

        context_image = (batch["context"]["image"] + 1) / 2
        if "target" in batch and "image" in batch["target"]:
            target_image = (batch["target"]["image"] + 1) / 2
            input_image = torch.cat([context_image, target_image], dim=1)
        else:
            input_image = context_image

        instance_mask = _cat_ctx_tgt(batch, "instance_mask")
        valid_mask = _cat_ctx_tgt(batch, "valid_mask")

        encoder_output, _ = self.model(
            input_image,
            self.global_step,
            instance_mask=instance_mask,
            valid_mask=valid_mask,
        )
        depth_dict = encoder_output.depth_dict or {}
        depth_dict_for_loss = dict(depth_dict)

        # ---- instance / FADA supervision ----
        if encoder_output.segvggt_prediction is not None:
            depth_dict_for_loss["segvggt_prediction"] = encoder_output.segvggt_prediction
        if instance_mask is not None:
            depth_dict_for_loss["instance_mask"] = instance_mask
        if valid_mask is not None:
            depth_dict_for_loss["instance_valid_mask"] = valid_mask

        # ---- geometry supervision ----
        if encoder_output.pred_pose_enc_list is not None:
            depth_dict_for_loss["pred_pose_enc_list"] = encoder_output.pred_pose_enc_list
        if self.geo_supervision == "teacher":
            geo_target = self._geo_target_from_teacher(input_image)
        else:
            geo_target = self._geo_target_from_gt(batch)
        if geo_target is not None:
            depth_dict_for_loss["segvggt_geo_target"] = geo_target

        with torch.amp.autocast("cuda", enabled=False):
            total_loss, loss_values = self.compute_and_log_losses(
                self.losses, None, batch, None, depth_dict_for_loss, self.global_step,
            )

        return self.finalize_training_step(total_loss, loss_values, batch)

    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        inst_mask = batch["context"].get("instance_mask")
        valid_mask = batch["context"].get("valid_mask")

        encoder_output, _ = self.model(
            (batch["context"]["image"] + 1) / 2,
            self.global_step,
            instance_mask=inst_mask,
            valid_mask=valid_mask,
        )
        depth_dict = encoder_output.depth_dict or {}
        pred = encoder_output.segvggt_prediction

        # log a coarse "how many queries fired" signal (1 - P(no-object) > 0.5)
        if pred is not None and pred.query_class_logits is not None:
            probs = pred.query_class_logits[0].float().softmax(-1)
            fired = int((1.0 - probs[:, -1] > 0.5).sum().item())
            self.log("val/queries_fired", float(fired))
        if "depth" in depth_dict and depth_dict["depth"] is not None:
            self.log("val/depth_mean", depth_dict["depth"].float().mean())

        if self.trainer.global_rank == 0 and batch_idx % 100 == 0:
            logger.info(
                "validation step %s; scene = %s; context = %s",
                self.global_step, batch["scene"], batch["context"]["index"].tolist(),
            )
            context_img = inverse_normalize(batch["context"]["image"][0])
            cols = [add_label(vcat(*context_img), "Context")]
            if "depth" in depth_dict and depth_dict["depth"] is not None:
                d = depth_dict["depth"].squeeze(-1)[0]
                cols.append(add_label(vcat(*vis_depth_map(d)), "Depth"))
            comparison = hcat(*cols)
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

    # ------------------------------------------------------------------ #
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if self.global_rank == 0 and batch_idx % 100 == 0:
            logger.info("Test step %s.", f"{batch_idx:0>6}")
        with self.benchmarker.time("encoder"):
            self.model((batch["context"]["image"] + 1) / 2, self.global_step)

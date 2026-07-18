"""IGGTWrapper -- training / validation / test for IGGT (encoder only).

Physics class path:
  batch["physics_target"]  → depth_dict["physics_target"]
  encoder.physics_prediction → depth_dict["physics_prediction"]

Physics property path:
  batch["physics_property_target"] → depth_dict["physics_property_target"]
  encoder.physics_property_prediction → depth_dict["physics_property_prediction"]

PhysGM path:
  batch["physgm_target"] → depth_dict["physgm_target"]
  encoder.physgm_prediction → depth_dict["physgm_prediction"]
"""

from __future__ import annotations

import json
import logging

import torch
from torch import nn

from src.dataset.physics.parsers import physgm_denormalize
from src.dataset.physics.types import PhysGMTarget, PhysicsPropertyTarget, PhysicsTarget
from src.dataset.types import BatchedExample
from src.global_cfg import get_cfg
from src.loss import Loss
from src.loss.loss_huber import HuberLoss
from src.misc.image_io import prep_image
from src.misc.step_tracker import StepTracker
from src.misc.utils import inverse_normalize, vis_depth_map
from src.visualization.annotation import add_label
from src.visualization.layout import add_border, hcat, vcat
from .base_wrapper import (
    BaseModelWrapper,
    OptimizerCfg,
    TestCfg,
    TrainCfg,
    visualize_instance_features,
)

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


class IGGTWrapper(BaseModelWrapper):
    """Training loop for IGGT (encoder only, instance / physics)."""

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
        infos = encoder_output.infos or {}

        self.log("train/scene_scale", infos.get("scene_scale", 0.0))

        depth_dict_for_loss = dict(depth_dict)

        if encoder_output.instance_feat_map is not None:
            depth_dict_for_loss["instance_feat_map"] = encoder_output.instance_feat_map
        if instance_mask is not None:
            depth_dict_for_loss["instance_mask"] = instance_mask
        if valid_mask is not None:
            depth_dict_for_loss["instance_valid_mask"] = valid_mask

        if encoder_output.physics_prediction is not None:
            depth_dict_for_loss["physics_prediction"] = encoder_output.physics_prediction
        if "physics_target" in batch:
            depth_dict_for_loss["physics_target"] = batch["physics_target"]

        if encoder_output.physics_property_prediction is not None:
            depth_dict_for_loss["physics_property_prediction"] = (
                encoder_output.physics_property_prediction
            )
        if "physics_property_target" in batch:
            depth_dict_for_loss["physics_property_target"] = batch[
                "physics_property_target"
            ]

        if encoder_output.physgm_prediction is not None:
            depth_dict_for_loss["physgm_prediction"] = encoder_output.physgm_prediction
        if "physgm_target" in batch:
            depth_dict_for_loss["physgm_target"] = batch["physgm_target"]

        with torch.amp.autocast("cuda", enabled=False):
            total_loss, loss_values = self.compute_and_log_losses(
                self.losses, None, batch, None, depth_dict_for_loss, self.global_step,
            )

        return self.finalize_training_step(total_loss, loss_values, batch)

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

        if "depth" in depth_dict:
            depth_pred_raw = depth_dict["depth"]
            if depth_pred_raw is not None and depth_pred_raw.numel() > 0:
                self.log("val/depth_mean", depth_pred_raw.mean())

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

            self._log_physics_predictions(encoder_output, batch)
            self._log_physics_property_predictions(encoder_output, batch)
            self._log_physgm_predictions(encoder_output, batch)

    @torch.no_grad()
    def _log_physics_predictions(self, encoder_output, batch: BatchedExample):
        """Log per-instance physics class predictions."""
        pred = encoder_output.physics_prediction
        if pred is None or pred.instance_logits is None or pred.instance_ids is None:
            return

        targets: list[PhysicsTarget] | None = batch.get("physics_target")
        class_names = pred.class_names
        if class_names is None and targets:
            class_names = targets[0].class_names

        results = []
        for b_idx, (logits_b, ids_b) in enumerate(
            zip(pred.instance_logits, pred.instance_ids, strict=True)
        ):
            lut = None
            if targets is not None and b_idx < len(targets):
                lut = targets[b_idx].label_lut

            for row in range(logits_b.shape[0]):
                probs = torch.softmax(logits_b[row].float(), dim=-1)
                pred_idx = int(probs.argmax().item())
                conf = float(probs[pred_idx].item())
                pred_name = (
                    class_names[pred_idx]
                    if class_names is not None and pred_idx < len(class_names)
                    else f"class_{pred_idx}"
                )

                gt_label = "N/A"
                id_int = int(ids_b[row].item())
                if lut is not None and id_int < lut.shape[0] and int(lut[id_int].item()) > 0:
                    gt_idx = int(lut[id_int].item()) - 1
                    gt_label = (
                        class_names[gt_idx]
                        if class_names is not None and gt_idx < len(class_names)
                        else f"class_{gt_idx}"
                    )

                results.append({
                    "id": id_int,
                    "predicted": pred_name,
                    "confidence": round(conf, 3),
                    "gt": gt_label,
                })

        if results:
            results.sort(key=lambda x: x["id"])
            scene = batch.get("scene", "?")
            logger.info(
                "[PhysPred step=%d scene=%s]\n%s",
                self.global_step, scene,
                json.dumps(results, indent=2, ensure_ascii=False),
            )

    @torch.no_grad()
    def _log_physics_property_predictions(self, encoder_output, batch: BatchedExample):
        """Log per-instance property mean / log_var summaries."""
        pred = encoder_output.physics_property_prediction
        if pred is None or pred.instance_values is None or pred.instance_ids is None:
            return

        targets: list[PhysicsPropertyTarget] | None = batch.get(
            "physics_property_target"
        )
        prop_names = pred.property_names or ()

        results = []
        for b_idx, (vals_b, ids_b) in enumerate(
            zip(pred.instance_values, pred.instance_ids, strict=True)
        ):
            tgt = (
                targets[b_idx]
                if targets is not None and b_idx < len(targets)
                else None
            )
            for row in range(vals_b.shape[0]):
                id_int = int(ids_b[row].item())
                entry = {"id": id_int, "pred": {}, "gt": {}}
                for p_i, name in enumerate(prop_names):
                    entry["pred"][name] = {
                        "mean": round(float(vals_b[row, p_i, 0].item()), 4),
                        "log_var": round(float(vals_b[row, p_i, 1].item()), 4),
                    }
                    if (
                        tgt is not None
                        and id_int < tgt.valid.shape[0]
                        and bool(tgt.valid[id_int].item())
                    ):
                        entry["gt"][name] = {
                            "mean": round(
                                float(tgt.mean_lut[id_int, p_i].item()), 4
                            ),
                            "log_var": round(
                                float(tgt.log_var_lut[id_int, p_i].item()), 4
                            ),
                        }
                results.append(entry)

        if results:
            results.sort(key=lambda x: x["id"])
            scene = batch.get("scene", "?")
            shown = results[:32]
            logger.info(
                "[PhysPropPred step=%d scene=%s n=%d (show %d)]\n%s",
                self.global_step,
                scene,
                len(results),
                len(shown),
                json.dumps(shown, indent=2, ensure_ascii=False),
            )

    @torch.no_grad()
    def _log_physgm_predictions(self, encoder_output, batch: BatchedExample):
        """Log per-instance PhysGM mu / var (z-scored) plus SI-unit means."""
        pred = encoder_output.physgm_prediction
        if pred is None or pred.instance_mu is None or pred.instance_ids is None:
            return

        targets: list[PhysGMTarget] | None = batch.get("physgm_target")
        prop_names = pred.property_names or ()

        results = []
        for b_idx, (mu_b, var_b, ids_b) in enumerate(
            zip(pred.instance_mu, pred.instance_var, pred.instance_ids, strict=True)
        ):
            tgt = (
                targets[b_idx]
                if targets is not None and b_idx < len(targets)
                else None
            )
            if mu_b.shape[0] == 0:
                continue
            mu_si = physgm_denormalize(mu_b)
            for row in range(mu_b.shape[0]):
                id_int = int(ids_b[row].item())
                entry = {"id": id_int, "pred": {}, "gt": {}}
                for p_i, name in enumerate(prop_names):
                    entry["pred"][name] = {
                        "z": round(float(mu_b[row, p_i].item()), 4),
                        "var": round(float(var_b[row, p_i].item()), 4),
                        "si": float(f"{mu_si[row, p_i].item():.4g}"),
                    }
                    if (
                        tgt is not None
                        and id_int < tgt.valid.shape[0]
                        and bool(tgt.valid[id_int].item())
                    ):
                        gt_si = physgm_denormalize(tgt.value_lut[id_int])
                        entry["gt"][name] = {
                            "z": round(float(tgt.value_lut[id_int, p_i].item()), 4),
                            "si": float(f"{gt_si[p_i].item():.4g}"),
                        }
                results.append(entry)

        if results:
            results.sort(key=lambda x: x["id"])
            scene = batch.get("scene", "?")
            shown = results[:32]
            logger.info(
                "[PhysGMPred step=%d scene=%s n=%d (show %d)]\n%s",
                self.global_step,
                scene,
                len(results),
                len(shown),
                json.dumps(shown, indent=2, ensure_ascii=False),
            )

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

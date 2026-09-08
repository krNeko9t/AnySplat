"""SegVGGTWrapper -- encoder-only training / validation for SegVGGT.

Mirrors :class:`IGGTWrapper` (no decoder / rendering).  All supervised frames live
in ``batch["context"]`` only (no NVS ``target``).  Supervision is routed to the
losses through ``depth_dict`` (the repo's established encoder-only convention):

  instance / FADA  (``LossSegVGGT``):
    encoder.segvggt_prediction  -> depth_dict['segvggt_prediction']
    batch instance_mask         -> depth_dict['instance_mask']
    (do NOT reuse depth valid_mask as instance_valid_mask; depth holes ≠ bad labels)

  per-query physics (``LossSegVGGT``'s ``lambda_phys`` term, opt-in):
    batch physgm_target         -> depth_dict['physgm_target']
    (the predictions themselves ride along inside segvggt_prediction)

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

Geometry is also reported as *read-only* validation metrics (``val/geo_*``, see
:meth:`SegVGGTWrapper._log_geo_drift`), which stay live even when
``segvggt_geo.weight`` is 0 and the geometry heads are frozen -- that is the only
way backbone drift into depth / pose becomes visible at all.

Physics is reported the same way (``val/phys_mae_*``, see
:mod:`src.evaluation.physics_metrics`): per-property MAE in reportable units for
the student, alongside a constant and a class-lookup baseline scored on the same
matched instances.  The training loss is a Gaussian NLL and says nothing
reportable on its own, so without these a run produces no readable signal about
the only thing the physics head is for.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.coord import se3_inv
from src.dataset.types import BatchedExample
from src.evaluation.instance_metrics import compute_instance_metrics_batch
from src.evaluation.physics_metrics import compute_physics_metrics_batch
from src.loss import Loss
from src.loss.loss_segvggt_geo import (
    LossSegVGGTGeo,
    LossSegVGGTGeoCfg,
    LossSegVGGTGeoCfgWrapper,
)
from src.loss.loss_huber import extri_intri_to_pose_encoding
from src.misc.step_tracker import StepTracker
from src.misc.utils import inverse_normalize, vis_depth_map
from src.visualization.annotation import add_label
from src.visualization.instance_viz import colorize_labels, make_color_lut
from src.visualization.layout import add_border, hcat, vcat
from src.misc.image_io import prep_image
from .base_wrapper import BaseModelWrapper, OptimizerCfg, TestCfg, TrainCfg

logger = logging.getLogger(__name__)

# Objectness cut-off shared by val metrics, the "queries fired" counter and the
# visualisation, so all three tell the same story. Matches the inference decoder's
# `1 - P(no-match)` criterion (scripts/segvggt_infer.py).
VAL_SCORE_THRESHOLD = 0.5


def _labels_to_rgb(labels_hw: np.ndarray, lut: np.ndarray, device) -> torch.Tensor:
    """[H, W] int labels -> [3, H, W] float tensor in [0, 1] (id 0 stays black)."""
    rgb = colorize_labels(labels_hw, lut, ignore_label=0)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0


def _colorize_gt_instances(inst_mask: torch.Tensor, device) -> list[torch.Tensor]:
    """[V, H, W] instance ids -> per-view [3, H, W] colour images (id 0 = black)."""
    gt = inst_mask.detach().cpu().numpy().astype(np.int32)
    ids = np.unique(gt)
    ids = ids[ids != 0]
    lut = make_color_lut(max(1, len(ids) + 1), seed=1)
    contiguous = np.zeros_like(gt, dtype=np.int32)
    for j, uid in enumerate(ids.tolist()):
        contiguous[gt == uid] = j + 1
    return [_labels_to_rgb(contiguous[v], lut, device) for v in range(gt.shape[0])]


@torch.no_grad()
def _colorize_pred_instances(
    query_masks: torch.Tensor,
    query_class_logits: torch.Tensor,
    hw: tuple[int, int],
    device,
) -> list[torch.Tensor] | None:
    """Per-view predicted instance map from the object queries.

    ``query_masks`` [Q, V, h, w] logits, ``query_class_logits`` [Q, C+1].  Queries
    whose objectness clears the threshold are upsampled to ``hw`` and each pixel takes
    the argmax query, provided that query's own mask probability also clears 0.5
    (otherwise the pixel is background).  Returns None when nothing fired.

    NOTE: the palette is independent of the GT one -- query index k has no reason to
    equal instance id k -- so compare *shapes*, not colours.
    """
    scores = 1.0 - query_class_logits.float().softmax(-1)[:, -1]
    fired = scores > VAL_SCORE_THRESHOLD
    if not bool(fired.any()):
        return None

    logits = query_masks[fired].float()                       # [F, V, h, w]
    logits = F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)
    probs = logits.sigmoid()                                  # [F, V, H, W]
    best = probs.argmax(dim=0)                                # [V, H, W]
    conf = probs.max(dim=0).values
    labels = torch.where(conf > 0.5, best + 1, torch.zeros_like(best))
    labels_np = labels.cpu().numpy().astype(np.int32)

    lut = make_color_lut(int(fired.sum()) + 1, seed=7)
    return [_labels_to_rgb(labels_np[v], lut, device) for v in range(labels_np.shape[0])]


def _ctx_views(batch: BatchedExample, key: str):
    """Read a per-view tensor from context only (SegVGGT has no NVS target)."""
    ctx = batch.get("context", {})
    return ctx.get(key)


def _assert_no_nvs_target(batch: BatchedExample) -> None:
    """SegVGGT/IGGT must not receive a non-empty NVS target (prevents fake-target regressions)."""
    tgt = batch.get("target")
    if tgt is None:
        return
    img = tgt.get("image") if isinstance(tgt, dict) else None
    if img is not None and img.shape[1] > 0:
        raise RuntimeError(
            "SegVGGT/IGGT received non-empty batch['target'] "
            f"(S={img.shape[1]}). Set view_sampler.num_target_views: 0 so all "
            "supervised frames live in context only."
        )


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
        # Read-only geometry-drift metric (validation only). Reuses the configured
        # LossSegVGGTGeo when there is one -- so its lambdas match the loss it
        # mirrors -- and falls back to a default instance when geometry is not in
        # the loss list at all, since the metric must survive either config.
        self._geo_metric = next(
            (l for l in losses if isinstance(l, LossSegVGGTGeo)), None
        ) or LossSegVGGTGeo(LossSegVGGTGeoCfgWrapper(LossSegVGGTGeoCfg()))

    # ------------------------------------------------------------------ #
    # geometry target
    # ------------------------------------------------------------------ #
    def _geo_target_from_gt(self, batch: BatchedExample) -> dict | None:
        """Build the geometry target from clean manifest GT (camera + depth).

        Cameras are expressed in the frame of context view 0
        (``c2w' = inv(c2w_0) @ c2w``) so pose supervision is relative, matching
        VGGT / multi-view gauge. Per-view depth is unchanged (camera-local).
        """
        extr = _ctx_views(batch, "extrinsics")   # [B, S, 4, 4] c2w
        intr = _ctx_views(batch, "intrinsics")   # [B, S, 3, 3] normalised
        if extr is None or intr is None:
            return None
        # Pose algebra must stay fp32: under bf16-mixed, matmul/@ is autocast
        # back to BF16 and linalg.inv / pose_enc would break or lose precision.
        image_hw = batch["context"]["image"].shape[-2:]
        with torch.autocast(device_type=extr.device.type, enabled=False):
            extr = extr.float()
            # First-camera canonicalization: view 0 -> identity.
            extr = se3_inv(extr[:, :1]) @ extr
            w2c = se3_inv(extr)  # camera-from-world
            pose_enc = extri_intri_to_pose_encoding(
                w2c[:, :, :3, :4], intr.float(), image_hw
            )

        depth = _ctx_views(batch, "depth")        # [B, S, H, W] or [B,S,H,W,1]
        valid = _ctx_views(batch, "valid_mask")
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

    @torch.no_grad()
    def _log_geo_drift(self, batch: BatchedExample, encoder_output, depth_dict: dict) -> None:
        """Log depth / pose error as *metrics only* -- never as loss.

        ``camera_head`` / ``depth_head`` are frozen and ``segvggt_geo.weight`` is 0,
        so nothing in the objective watches geometry: unfreezing the backbone (LoRA)
        can move it and no number would say so.  These are that missing second
        column next to segmentation / physics.  Read-only by construction: no grad,
        no ``requires_grad`` touched, the freeze lock is unchanged.

        The ruler is always manifest GT, even when training distils geometry from
        the VGGT teacher -- drift is only meaningful against something fixed.
        """
        geo_target = self._geo_target_from_gt(batch)
        if geo_target is None:
            return
        d = dict(depth_dict)
        d["segvggt_geo_target"] = geo_target
        if encoder_output.pred_pose_enc_list is not None:
            d["pred_pose_enc_list"] = encoder_output.pred_pose_enc_list
        with torch.amp.autocast("cuda", enabled=False):
            m = self._geo_metric.metrics(d)
        # `sorted`, and it is load-bearing on every val log loop in this class.
        # `sync_dist=True` makes Lightning all-reduce each metric across ranks,
        # and it walks its results in *insertion* order -- so if two ranks
        # insert the same names in two different orders, the collectives pair up
        # mismatched metrics and each rank gets back someone else's number.
        # Iterating a `set` of strings does exactly that (string hashing is
        # salted per process).  Measured on a 2-GPU smoke run before this line
        # existed: nu's constant baseline came back 0.5285 where the truth was
        # 0.0498, with every other tag plausible-looking and wrong.
        for key, value in sorted(m.items()):
            self.log(f"val/{key}", float(value), sync_dist=True)

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        batch = self.combine_batches(batch)
        batch: BatchedExample = self.data_shim(batch)
        _assert_no_nvs_target(batch)

        input_image = (batch["context"]["image"] + 1) / 2
        instance_mask = _ctx_views(batch, "instance_mask")

        encoder_output, _ = self.model(
            input_image,
            self.global_step,
        )
        depth_dict = encoder_output.depth_dict or {}
        depth_dict_for_loss = dict(depth_dict)

        # ---- instance / FADA supervision ----
        if encoder_output.segvggt_prediction is not None:
            depth_dict_for_loss["segvggt_prediction"] = encoder_output.segvggt_prediction
        if instance_mask is not None:
            depth_dict_for_loss["instance_mask"] = instance_mask
        # Do not set instance_valid_mask from depth valid_mask (I1). Depth valid
        # only belongs in segvggt_geo_target via _geo_target_from_gt.

        # ---- per-query physics supervision (LossSegVGGT's lambda_phys term) ----
        # Per-scene instance-id -> property LUTs from the dataset physics parser;
        # collate keeps them as a list of PhysGMTarget. Mirrors IGGTWrapper.
        if "physgm_target" in batch:
            depth_dict_for_loss["physgm_target"] = batch["physgm_target"]

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
        _assert_no_nvs_target(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        inst_mask = batch["context"].get("instance_mask")

        encoder_output, _ = self.model(
            (batch["context"]["image"] + 1) / 2,
            self.global_step,
        )
        depth_dict = encoder_output.depth_dict or {}
        pred = encoder_output.segvggt_prediction

        # log a coarse "how many queries fired" signal (1 - P(no-match) > 0.5)
        if pred is not None and pred.query_class_logits is not None:
            probs = pred.query_class_logits[0].float().softmax(-1)
            fired = int((1.0 - probs[:, -1] > VAL_SCORE_THRESHOLD).sum().item())
            self.log("val/queries_fired", float(fired), sync_dist=True)
        if "depth" in depth_dict and depth_dict["depth"] is not None:
            self.log("val/depth_mean", depth_dict["depth"].float().mean(), sync_dist=True)

        # ---- read-only geometry drift (val/geo_*) --------------------------
        self._log_geo_drift(batch, encoder_output, depth_dict)

        # ---- instance-segmentation metrics --------------------------------
        # The only in-flight signal that distinguishes "not converged yet" from
        # "the training objective is not being optimised at all": val/matched_iou_mean
        # is exactly what the mask loss maximises and should climb towards >0.5.
        # See src/evaluation/instance_metrics.py for what each number means.
        if (
            pred is not None
            and pred.query_masks is not None
            and pred.query_class_logits is not None
            and inst_mask is not None
        ):
            metrics = compute_instance_metrics_batch(
                pred.query_masks,
                pred.query_class_logits,
                inst_mask,
                None,  # do not crop instance GT with depth valid_mask
                score_threshold=VAL_SCORE_THRESHOLD,
            )
            for key, value in sorted(metrics.items()):   # order matters: see _log_geo_drift
                self.log(f"val/{key}", float(value), sync_dist=True)

        # ---- physics: student vs the class-lookup baseline -----------------
        # The map's destination is "student approaches teacher", and ticket 03
        # fixes the reference frame: the teacher is not a row (the student's
        # labels *are* the teacher), so what gets reported is the student's
        # position relative to a training-split class lookup table.  Scored on
        # the same IoU-optimal match as val/matched_iou_mean, so segmentation
        # and physics describe the same instances.  Read-only, no grad.
        if (
            pred is not None
            and pred.query_masks is not None
            and pred.query_phys_mu is not None
            and inst_mask is not None
            and "physgm_target" in batch
        ):
            phys_metrics = compute_physics_metrics_batch(
                pred.query_masks,
                pred.query_phys_mu,
                inst_mask,
                batch["physgm_target"],
                None,  # same as the instance metrics: do not crop GT with valid_mask
            )
            for key, value in sorted(phys_metrics.items()):  # order matters: see _log_geo_drift
                self.log(f"val/{key}", float(value), sync_dist=True)

        if self.trainer.global_rank == 0 and batch_idx % 100 == 0:
            logger.info(
                "validation step %s; scene = %s; context = %s",
                self.global_step, batch["scene"], batch["context"]["index"].tolist(),
            )
            context_img = inverse_normalize(batch["context"]["image"][0])
            cols = [add_label(vcat(*context_img), "Context")]
            if inst_mask is not None:
                gt_cols = _colorize_gt_instances(inst_mask[0], context_img.device)
                cols.append(add_label(vcat(*gt_cols), "GT inst"))
            if (
                pred is not None
                and pred.query_masks is not None
                and pred.query_class_logits is not None
            ):
                pred_cols = _colorize_pred_instances(
                    pred.query_masks[0],
                    pred.query_class_logits[0],
                    context_img.shape[-2:],
                    context_img.device,
                )
                # palettes are independent of the GT one -- compare shapes, not colours
                if pred_cols is not None:
                    cols.append(add_label(vcat(*pred_cols), "Pred inst"))
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
    def on_validation_epoch_end(self) -> None:
        """Print the numbers the curves are actually made of -- nothing else.

        The console lines used to sit inside ``validation_step`` behind
        ``batch_idx % 100 == 0``, and a rank sees only ~10 val batches, so every
        printed row was *batch 0 alone*: 2 scenes, ~36 instances.  At the last
        validation of arm (a) that row said E/clut = 1.447 while the aggregate
        was 0.952 -- opposite signs of the same comparison, and this round it
        was read as "the student is worse than the constant baseline" before
        the tfevents said otherwise (ticket 13.3).  Lightning has already
        reduced these over batches and (``sync_dist=True``) over ranks by the
        time this hook runs, so what is printed is exactly what TensorBoard gets.
        """
        if self.trainer.global_rank != 0:
            return
        metrics = self.trainer.callback_metrics

        def m(key: str) -> float:
            value = metrics.get(f"val/{key}")
            return float("nan") if value is None else float(value)

        logger.info(
            "[val step=%d] matched_iou=%.4f best_iou_per_gt=%.4f "
            "ap50=%.4f ap25=%.4f n_gt=%.1f n_fired=%.1f area=%.4f",
            self.global_step, m("matched_iou_mean"), m("best_iou_per_gt_mean"),
            m("ap50"), m("ap25"), m("n_gt"), m("n_fired"), m("mask_area_mean"),
        )
        logger.info(
            "[val phys step=%d n=%.1f] log10 E %.4f (const %.4f, clut %.4f)  "
            "log10 rho %.4f (const %.4f, clut %.4f)  nu %.4f (const %.4f, clut %.4f)",
            self.global_step, m("phys_n_matched"),
            m("phys_mae_log10_youngs_modulus"),
            m("phys_mae_log10_youngs_modulus_const"),
            m("phys_mae_log10_youngs_modulus_clut"),
            m("phys_mae_log10_density"),
            m("phys_mae_log10_density_const"),
            m("phys_mae_log10_density_clut"),
            m("phys_mae_raw_poisson_ratio"),
            m("phys_mae_raw_poisson_ratio_const"),
            m("phys_mae_raw_poisson_ratio_clut"),
        )
        # ...and the same three properties in the caliber the paper table uses
        # (instance-weighted pooled, ticket 13.2): mean(<tag>_xn) / mean(n).
        n = m("phys_n_matched")

        def pooled(key: str) -> float:
            return m(f"{key}_xn") / n if n else float("nan")

        logger.info(
            "[val phys pooled step=%d] log10 E %.4f (const %.4f, clut %.4f)  "
            "log10 rho %.4f (const %.4f, clut %.4f)  nu %.4f (const %.4f, clut %.4f)",
            self.global_step,
            pooled("phys_mae_log10_youngs_modulus"),
            pooled("phys_mae_log10_youngs_modulus_const"),
            pooled("phys_mae_log10_youngs_modulus_clut"),
            pooled("phys_mae_log10_density"),
            pooled("phys_mae_log10_density_const"),
            pooled("phys_mae_log10_density_clut"),
            pooled("phys_mae_raw_poisson_ratio"),
            pooled("phys_mae_raw_poisson_ratio_const"),
            pooled("phys_mae_raw_poisson_ratio_clut"),
        )
        logger.info(
            "[val geo step=%d] camera=%.4f (T=%.4f R=%.4f fl=%.4f) "
            "depth=%.3f depth_rel=%.4f",
            self.global_step, m("geo_camera"), m("geo_camera_T_last"),
            m("geo_camera_R_last"), m("geo_camera_fl_last"),
            m("geo_depth"), m("geo_depth_rel"),
        )

    # ------------------------------------------------------------------ #
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _assert_no_nvs_target(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if self.global_rank == 0 and batch_idx % 100 == 0:
            logger.info("Test step %s.", f"{batch_idx:0>6}")
        with self.benchmarker.time("encoder"):
            self.model((batch["context"]["image"] + 1) / 2, self.global_step)

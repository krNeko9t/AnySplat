"""AnySplatWrapper -- training / validation / test for AnySplat (encoder + decoder)."""

from __future__ import annotations

import gc
import logging
from typing import Optional, Any

import numpy as np
import torch
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_distill import DistillLoss
from ..loss.loss_huber import HuberLoss
from ..misc.cam_utils import rotation_6d_to_matrix
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.layout import add_border, hcat, vcat
from .base_wrapper import (
    BaseModelWrapper,
    OptimizerCfg,
    TestCfg,
    TrainCfg,
    _write_video_async,
    log_video,
    visualize_instance_features,
)
from .decoder.decoder import DecoderOutput
from .encoder.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply
from src.utils.point import get_normal_map

logger = logging.getLogger(__name__)


def _align_gt_trajectory_to_pred(extrinsics, pred_ext, gt_ext):
    pred_0 = pred_ext[0, 0]
    pred_1 = pred_ext[0, min(1, pred_ext.shape[1] - 1)]
    gt_0 = gt_ext[0, 0]
    gt_1 = gt_ext[0, min(1, gt_ext.shape[1] - 1)]

    scale = (pred_1[:3, 3] - pred_0[:3, 3]).norm() / (
        (gt_1[:3, 3] - gt_0[:3, 3]).norm() + 1e-8
    )
    R_align = pred_0[:3, :3] @ gt_0[:3, :3].T
    t_align = pred_0[:3, 3] - scale * (gt_0[:3, 3])

    out = extrinsics.clone()
    out[..., :3, :3] = R_align @ extrinsics[..., :3, :3]
    out[..., :3, 3] = scale * extrinsics[..., :3, 3] + t_align
    return out


class AnySplatWrapper(BaseModelWrapper):
    """Training loop for AnySplat (encoder + Gaussian decoder)."""

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

        if self.model.encoder.pred_pose:
            self.loss_pose = HuberLoss(
                alpha=self.train_cfg.pose_loss_alpha,
                delta=self.train_cfg.pose_loss_delta,
            )

        if self.model.encoder.distill:
            self.loss_distill = DistillLoss(
                delta=self.train_cfg.pose_loss_delta,
                weight_pose=self.train_cfg.weight_pose,
                weight_depth=self.train_cfg.weight_depth,
                weight_normal=self.train_cfg.weight_normal,
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

        encoder_output, output_all = self.model(input_image, self.global_step, visualization_dump=None)
        gaussians = encoder_output.gaussians
        pred_pose_enc_list = encoder_output.pred_pose_enc_list
        depth_dict = encoder_output.depth_dict or {}
        infos = encoder_output.infos
        distill_infos = encoder_output.distill_infos

        def _slice_views(x: Any, v: int):
            if torch.is_tensor(x) and x.ndim >= 2 and x.shape[1] >= v:
                return x[:, :v]
            return x

        distill_infos_ctx = None
        if distill_infos is not None:
            distill_infos_ctx = {k: _slice_views(v, v_ctx) for k, v in distill_infos.items()}
        depth_dict_ctx = {k: _slice_views(v, v_ctx) for k, v in depth_dict.items()}

        using_index = torch.arange(v_ctx, device=input_image.device)
        batch["using_index"] = using_index

        output = DecoderOutput(
            color=output_all.color[:, :v_ctx],
            depth=output_all.depth[:, :v_ctx],
            alpha=output_all.alpha[:, :v_ctx],
            lod_rendering=output_all.lod_rendering,
        )
        target_gt = (batch["context"]["image"] + 1) / 2
        self.log("train/scene_scale", infos["scene_scale"])
        self.log("train/voxelize_ratio", infos["voxelize_ratio"])

        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        depth_pred = depth_dict_ctx.get("depth")
        conf_mask = distill_infos_ctx.get("conf_mask") if distill_infos_ctx is not None else None
        if depth_pred is not None and conf_mask is not None:
            consis_absrel = abs_relative_difference(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_pred[:, :v_ctx].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(conf_mask[:, :v_ctx], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_absrel", consis_absrel.mean())
            consis_delta1 = delta1_acc(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_pred[:, :v_ctx].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(conf_mask[:, :v_ctx], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_delta1", consis_delta1.mean())
        else:
            self.log("train/consis_absrel", torch.tensor(0.0, device=self.device))
            self.log("train/consis_delta1", torch.tensor(0.0, device=self.device))

        depth_dict_ctx["distill_infos"] = distill_infos_ctx
        if encoder_output.instance_feat_map is not None:
            depth_dict_ctx["instance_feat_map"] = encoder_output.instance_feat_map
        if "context" in batch and "instance_mask" in batch["context"] and "target" in batch and "instance_mask" in batch["target"]:
            depth_dict_ctx["instance_mask"] = torch.cat(
                [batch["context"]["instance_mask"], batch["target"]["instance_mask"]], dim=1,
            )
            if "instance_feat_map" in depth_dict_ctx:
                assert depth_dict_ctx["instance_feat_map"].shape[1] == depth_dict_ctx["instance_mask"].shape[1]
        if "context" in batch and "valid_mask" in batch["context"] and "target" in batch and "valid_mask" in batch["target"]:
            depth_dict_ctx["instance_valid_mask"] = torch.cat(
                [batch["context"]["valid_mask"], batch["target"]["valid_mask"]], dim=1,
            )

        with torch.amp.autocast("cuda", enabled=False):
            total_loss, loss_values = self.compute_and_log_losses(
                self.losses, output, batch, gaussians, depth_dict_ctx, self.global_step,
            )

            if depth_dict_ctx is not None and "depth" in get_cfg()["loss"].keys() and self.train_cfg.cxt_depth_weight > 0:
                depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
                depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
                loss_depth = depth_loss_fn(
                    depth_dict_ctx["depth_map"], depth_dict_ctx["depth_conf"],
                    batch, cxt_depth_weight=self.train_cfg.cxt_depth_weight,
                )
                self.log("loss/ctx_depth", loss_depth)
                try:
                    loss_values["ctx_depth"] = float(loss_depth.detach().item())
                except Exception:
                    pass
                total_loss = total_loss + loss_depth

            if distill_infos_ctx is not None and len(distill_infos_ctx) > 0:
                loss_distill_list = self.loss_distill(distill_infos_ctx, pred_pose_enc_list, output, batch)
                self.log("loss/distill", loss_distill_list["loss_distill"])
                self.log("loss/distill_pose", loss_distill_list["loss_pose"])
                self.log("loss/distill_depth", loss_distill_list["loss_depth"])
                self.log("loss/distill_normal", loss_distill_list["loss_normal"])
                for k, v in loss_distill_list.items():
                    if torch.is_tensor(v):
                        try:
                            loss_values[k] = float(v.detach().item())
                        except Exception:
                            pass
                total_loss = total_loss + loss_distill_list["loss_distill"]

        return self.finalize_training_step(total_loss, loss_values, batch)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if self.global_rank == 0 and batch_idx % 100 == 0:
            logger.info("Test step %s.", f"{batch_idx:0>6}")

        with self.benchmarker.time("encoder"):
            encoder_output = self.model.encoder(
                (batch["context"]["image"] + 1) / 2, self.global_step,
            )
            gaussians = encoder_output.gaussians

        if self.test_cfg.align_pose:
            output = self._test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.model.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)
            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                "lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                "ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                "psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, ["ours"], overlap_tag=overlap_tag)

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video([a for a in output.color[0]], path / "video" / f"{scene}_frame_{frame_str}.mp4")
        if self.test_cfg.save_compare:
            context_img = inverse_normalize(batch["context"]["image"][0])
            rgb_gt = batch["target"]["image"][0]
            rgb_pred = output.color[0]
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

    def _test_step_align(self, batch, gaussians):
        self.model.encoder.eval()
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        output_c2ws = batch["target"]["extrinsics"]
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 6], requires_grad=True, device=output_c2ws.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=output_c2ws.device))
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).to(output_c2ws))
            opt_params = [
                {"params": [cam_rot_delta], "lr": 0.005},
                {"params": [cam_trans_delta], "lr": 0.005},
            ]
            pose_optimizer = torch.optim.Adam(opt_params)
            extrinsics = output_c2ws.clone()
            with self.benchmarker.time("optimize"):
                for _ in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()
                    rot = rotation_6d_to_matrix(cam_rot_delta + self.identity.expand(b, v, -1))
                    transform = torch.eye(4, device=extrinsics.device).repeat((b, v, 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = cam_trans_delta
                    new_extrinsics = torch.matmul(extrinsics, transform)
                    output = self.model.decoder.forward(
                        gaussians, new_extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                    total_loss = sum(
                        loss_fn.forward(output, batch, gaussians, self.global_step)
                        for loss_fn in self.losses
                    )
                    total_loss.backward()
                    pose_optimizer.step()

        return self.model.decoder.forward(
            gaussians, new_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        visualization_dump = {}

        encoder_output, output = self.model(
            (batch["context"]["image"] + 1) / 2,
            self.global_step,
            visualization_dump=visualization_dump,
        )
        gaussians = encoder_output.gaussians
        depth_dict = encoder_output.depth_dict
        distill_infos = encoder_output.distill_infos
        infos = encoder_output.infos

        GS_num = infos["voxelize_ratio"] * (h * w * v)
        self.log("val/GS_num", GS_num)

        rgb_pred = output.color[0].float()
        depth_pred = vis_depth_map(output.depth[0])

        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        rgb_gt = (batch["context"]["image"][0].float() + 1) / 2
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log("val/psnr", psnr)
        lpips_val = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log("val/lpips", lpips_val)
        ssim_val = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log("val/ssim", ssim_val)

        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict["depth"].squeeze(-1), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_absrel", consis_absrel.mean())

        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict["depth"].squeeze(-1), "b v h w -> (b v) h w"),
            valid_mask=rearrange(
                torch.ones_like(output.depth, dtype=torch.bool), "b v h w -> (b v) h w",
            ),
        )
        self.log("val/consis_delta1", consis_delta1.mean())

        diff_map = torch.abs(output.depth - depth_dict["depth"].squeeze(-1))
        try:
            consis_mse = diff_map[distill_infos["conf_mask"]].mean()
        except Exception:
            consis_mse = torch.tensor(0.0, device=diff_map.device)
        self.log("val/consis_mse", consis_mse)

        if self.trainer.global_rank == 0:
            logger.info(
                "validation step %s; scene = %s; context = %s",
                self.global_step, batch["scene"], batch["context"]["index"].tolist(),
            )
            if batch_idx % 100 == 0:
                context_img = inverse_normalize(batch["context"]["image"][0])
                colored_diff_map = vis_depth_map(
                    diff_map[0],
                    near=torch.tensor(1e-4, device=diff_map.device),
                    far=torch.tensor(1.0, device=diff_map.device),
                )
                model_depth_pred = vis_depth_map(depth_dict["depth"].squeeze(-1)[0])
                render_normal = (
                    get_normal_map(
                        output.depth.flatten(0, 1),
                        batch["context"]["intrinsics"].flatten(0, 1),
                    ).permute(0, 3, 1, 2) + 1
                ) / 2.0
                pred_normal = (
                    get_normal_map(
                        depth_dict["depth"].flatten(0, 1).squeeze(-1),
                        batch["context"]["intrinsics"].flatten(0, 1),
                    ).permute(0, 3, 1, 2) + 1
                ) / 2.0

                comparison = hcat(
                    add_label(vcat(*context_img), "Context"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_pred), "Target (Prediction)"),
                    add_label(vcat(*depth_pred), "Depth (Prediction)"),
                    add_label(vcat(*model_depth_pred), "Depth (VGGT Prediction)"),
                    add_label(vcat(*render_normal), "Normal (Prediction)"),
                    add_label(vcat(*pred_normal), "Normal (VGGT Prediction)"),
                    add_label(vcat(*colored_diff_map), "Diff Map"),
                )
                comparison = torch.nn.functional.interpolate(
                    comparison.unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False,
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

                if self.encoder_visualizer is not None:
                    for k, image in self.encoder_visualizer.visualize(batch["context"], self.global_step).items():
                        self.logger.log_image(k, [prep_image(image)], step=self.global_step)

                self.render_video_interpolation(batch)
                self.render_video_wobble(batch)
                if self.train_cfg.extended_visualization:
                    self.render_video_interpolation_exaggerated(batch)

    # ------------------------------------------------------------------
    # Video rendering
    # ------------------------------------------------------------------

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        use_gt = getattr(self.train_cfg, "video_use_gt_trajectory", False)
        if use_gt:
            gt_ext = batch["context"]["extrinsics"]
            gt_intr = batch["context"]["intrinsics"]

            def trajectory_fn(t, pred_extrinsics, pred_intrinsics):
                _, v, _, _ = gt_ext.shape
                if v < 2:
                    return None, None
                delta = (gt_ext[:, 0, :3, 3] - gt_ext[:, 1, :3, 3]).norm(dim=-1)
                extrinsics = generate_wobble(gt_ext[:, 0], delta * 0.25, t)
                extrinsics = _align_gt_trajectory_to_pred(extrinsics, pred_extrinsics, gt_ext)
                intrinsics = repeat(gt_intr[:, 0], "b i j -> b v i j", v=t.shape[0])
                return extrinsics, intrinsics
        else:
            def trajectory_fn(t, pred_extrinsics, pred_intrinsics):
                _, v, _, _ = pred_extrinsics.shape
                if v < 2:
                    return None, None
                delta = (pred_extrinsics[:, 0, :3, 3] - pred_extrinsics[:, 1, :3, 3]).norm(dim=-1)
                extrinsics = generate_wobble(pred_extrinsics[:, 0], delta * 0.25, t)
                intrinsics = repeat(pred_intrinsics[:, 0], "b i j -> b v i j", v=t.shape[0])
                return extrinsics, intrinsics

        self._render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        use_gt = getattr(self.train_cfg, "video_use_gt_trajectory", False)
        if use_gt:
            gt_ext = batch["context"]["extrinsics"]
            gt_intr = batch["context"]["intrinsics"]

            def trajectory_fn(t, pred_extrinsics, pred_intrinsics):
                _, v, _, _ = gt_ext.shape
                idx1 = min(1, v - 1)
                extrinsics = interpolate_extrinsics(gt_ext[0, 0], gt_ext[0, idx1], t)
                extrinsics = _align_gt_trajectory_to_pred(extrinsics, pred_extrinsics, gt_ext)
                intrinsics = interpolate_intrinsics(gt_intr[0, 0], gt_intr[0, idx1], t)
                return extrinsics[None], intrinsics[None]
        else:
            def trajectory_fn(t, pred_extrinsics, pred_intrinsics):
                _, v, _, _ = pred_extrinsics.shape
                idx1 = min(1, v - 1)
                extrinsics = interpolate_extrinsics(pred_extrinsics[0, 0], pred_extrinsics[0, idx1], t)
                intrinsics = interpolate_intrinsics(pred_intrinsics[0, 0], pred_intrinsics[0, idx1], t)
                return extrinsics[None], intrinsics[None]

        self._render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        def trajectory_fn(t, pred_extrinsics, pred_intrinsics):
            _, v, _, _ = pred_extrinsics.shape
            if v < 2:
                return None, None
            delta = (pred_extrinsics[:, 0, :3, 3] - pred_extrinsics[:, 1, :3, 3]).norm(dim=-1)
            tf = generate_wobble_transformation(delta * 0.5, t, 5, scale_radius_with_t=False)
            extrinsics = interpolate_extrinsics(pred_extrinsics[0, 0], pred_extrinsics[0, 1], t * 5 - 2)
            intrinsics = interpolate_intrinsics(pred_intrinsics[0, 0], pred_intrinsics[0, 1], t * 5 - 2)
            return extrinsics @ tf, intrinsics[None]

        self._render_video_generic(
            batch, trajectory_fn, "interpolation_exagerrated",
            num_frames=300, smooth=False, loop_reverse=False,
        )

    @rank_zero_only
    def _render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        try:
            encoder_output = self.model.encoder((batch["context"]["image"] + 1) / 2, self.global_step)
            gaussians = encoder_output.gaussians
            pred_ext = encoder_output.pred_context_pose["extrinsic"]
            pred_intr = encoder_output.pred_context_pose["intrinsic"]

            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
            if smooth:
                t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

            extrinsics, intrinsics = trajectory_fn(t, pred_ext, pred_intr)
            if extrinsics is None:
                return

            _, _, _, h, w = batch["context"]["image"].shape
            near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
            far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
            output = self.model.decoder.forward(
                gaussians, extrinsics, intrinsics, near, far, (h, w), "depth",
            )
            images = [
                vcat(rgb, depth)
                for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
            ]

            video = torch.stack(images)
            video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            if loop_reverse:
                video = pack([video, video[::-1][1:-1]], "* c h w")[0]

            log_video(self.logger, f"video/{name}", video, self.global_step)
        except Exception as exc:
            logger.warning("render_video_generic(%s) failed: %s", name, exc)


# Backward-compatible alias
ModelWrapper = AnySplatWrapper

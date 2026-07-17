"""BaseModelWrapper -- shared Lightning infrastructure for all architectures.

Subclasses must implement ``training_step``, ``validation_step``, and
``test_step``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gc
import logging
import math
import threading
from typing import Literal, Optional, Protocol, runtime_checkable, Any

import numpy as np
import moviepy.editor as mpy
import torch
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.tb_logger import TBLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize
from ..visualization.annotation import add_label
from ..visualization.instance_viz import (
    cluster_instance_embeddings,
    colorize_labels,
    knn_smooth_instance_features,
    make_color_lut,
    pca_visualize_embeddings,
)
from ..visualization.layout import add_border, hcat, vcat
from .decoder.decoder import DepthRenderingMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses (shared by all wrappers)
# ---------------------------------------------------------------------------


@dataclass
class ParamGroupCfg:
    """Learning rate group: params whose name contains any keyword get lr * lr_multiplier."""
    keywords: list[str]
    lr_multiplier: float


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float
    new_param_keywords: list[str] = field(
        default_factory=lambda: ["gaussian_param_head", "interm"]
    )
    # Single source of truth for freezing when non-empty: params matching any
    # keyword are frozen, all others unfrozen. Applied in BaseWrapper.setup().
    freeze_keywords: list[str] = field(default_factory=list)
    # Declarative LR groups. First match wins. Unmatched params use backbone_lr_multiplier.
    # When non-empty, overrides new_param_keywords logic.
    param_groups: list[ParamGroupCfg] = field(default_factory=list)


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0
    video_use_gt_trajectory: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_video_async(path: Path, tensor, fps: int = 30) -> None:
    """Encode and write a video file in a daemon thread."""
    def _worker():
        try:
            clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
            clip.write_videofile(str(path), logger=None)
        except Exception as exc:
            logger.warning("Background video write to %s failed: %s", path, exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def log_video(pl_logger, tag: str, video: np.ndarray, global_step: int) -> None:
    """Log a video tensor to the active logger (W&B / TensorBoard / local)."""
    if isinstance(pl_logger, WandbLogger):
        try:
            import wandb
            wandb.log({tag: wandb.Video(video[None], fps=30, format="mp4")})
        except Exception:
            pass
    elif isinstance(pl_logger, TBLogger):
        vid_tensor = torch.from_numpy(video).float() / 255.0
        if vid_tensor.dim() == 4:
            vid_tensor = vid_tensor.unsqueeze(0)
        pl_logger.experiment.add_video(
            tag=tag, vid_tensor=vid_tensor, global_step=global_step, fps=30,
        )
    elif isinstance(pl_logger, LocalLogger):
        vid_dir = LOG_PATH / tag
        vid_dir.mkdir(exist_ok=True, parents=True)
        vid_path = vid_dir / f"{global_step:0>6}.mp4"
        frames = [video[i].transpose(1, 2, 0) for i in range(video.shape[0])]
        _write_video_async(vid_path, frames)


def visualize_instance_features(
    encoder_output,
    batch: BatchedExample,
    context_img: Tensor,
    depth_dict: dict,
    pl_logger,
    global_step: int,
) -> None:
    """Build and log the instance head comparison image (shared by all wrappers).

    Aligns with iggt_idmap.py: 3D KNN smoothing (when world_points available)
    and PCA on L2-normalized + centered features.
    """
    if encoder_output.instance_feat_map is None:
        return
    if "instance_mask" not in batch.get("context", {}):
        return

    feat_map = encoder_output.instance_feat_map[0].float()
    feat_map = knn_smooth_instance_features(feat_map, depth_dict, k=20)
    V, N, H, W = feat_map.shape
    valid = depth_dict.get("conf_valid_mask")
    valid_vhw = valid[0] if valid is not None else None
    k_cluster = 20

    gt_np = batch["context"]["instance_mask"][0].detach().cpu().numpy().astype(np.int32)
    unique_gt = np.unique(gt_np)
    unique_gt = unique_gt[unique_gt != 0]
    if len(unique_gt) > 0:
        k_cluster = max(2, min(k_cluster, len(unique_gt) + 2))

    pred_labels = cluster_instance_embeddings(feat_map, valid_vhw, k=k_cluster, seed=0)

    gt_contig = np.zeros_like(gt_np, dtype=np.int32)
    id_to_contig = {int(i): (j + 1) for j, i in enumerate(unique_gt.tolist())}
    for i, j in id_to_contig.items():
        gt_contig[gt_np == i] = int(j)

    pred_lut = make_color_lut(max(1, k_cluster + 1), seed=0)
    gt_lut = make_color_lut(max(1, len(unique_gt) + 1), seed=1)
    dev = context_img.device

    pred_colored = [
        torch.from_numpy(colorize_labels(pred_labels[vi], pred_lut, ignore_label=0))
        .permute(2, 0, 1).float().to(dev) / 255.0
        for vi in range(V)
    ]
    gt_colored = [
        torch.from_numpy(colorize_labels(gt_contig[vi], gt_lut, ignore_label=0))
        .permute(2, 0, 1).float().to(dev) / 255.0
        for vi in range(V)
    ]
    pca_vis = pca_visualize_embeddings(feat_map, valid_vhw)
    pca_list = [pca_vis[vi] for vi in range(V)]
    ctx_list = [context_img[vi] for vi in range(V)]

    instance_comparison = hcat(
        add_label(vcat(*ctx_list), "Context"),
        add_label(vcat(*pred_colored), "Clustered Instance"),
        add_label(vcat(*gt_colored), "GT ID Map"),
        add_label(vcat(*pca_list), "PCA Instance"),
    )
    instance_comparison = torch.nn.functional.interpolate(
        instance_comparison.unsqueeze(0),
        scale_factor=0.5,
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    pl_logger.log_image(
        "instance_comparison",
        [prep_image(add_border(instance_comparison))],
        step=global_step,
        caption=batch["scene"],
    )


# ---------------------------------------------------------------------------
# BaseModelWrapper
# ---------------------------------------------------------------------------


class BaseModelWrapper(LightningModule):
    """Shared infrastructure: optimizer, callbacks, logging, metrics."""

    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)
        self.benchmarker = Benchmarker()

    # ---- Epoch / batch callbacks ----

    def on_train_epoch_start(self) -> None:
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        try:
            step = int(self.global_step)
        except Exception:
            step = -1
        if step >= 0 and (step % 500 == 0):
            logger.info(
                "Train batch start step=%s batch_idx=%s rank=%s",
                step, batch_idx, self.trainer.global_rank,
            )

    def on_validation_epoch_start(self) -> None:
        logger.info("Validation epoch start on rank %s", self.trainer.global_rank)
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_batch_start(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        logger.info(
            "Validation batch start step=%s batch_idx=%s rank=%s",
            int(self.global_step), batch_idx, self.trainer.global_rank,
        )

    def on_validation_epoch_end(self) -> None:
        logger.info(
            "Validation epoch end step=%s rank=%s",
            int(self.global_step), self.trainer.global_rank,
        )

    def on_after_backward(self):
        total_norm = 0.0
        counter = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        if counter > 0:
            total_norm = (total_norm / counter) ** 0.5
        self.log("loss/grad_norm", total_norm)

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(self.test_cfg.output_path / name / "peak_memory.json")
        self.benchmarker.summarize()

    # ---- Shared helpers ----

    def combine_batches(self, batch) -> dict:
        """Merge batches from multiple dataloaders."""
        if not isinstance(batch, list):
            return batch
        batch_combined = None
        for batch_per_dl in batch:
            if batch_combined is None:
                batch_combined = batch_per_dl
            else:
                for k in batch_combined.keys():
                    if isinstance(batch_combined[k], list):
                        batch_combined[k] += batch_per_dl[k]
                    elif isinstance(batch_combined[k], dict):
                        for kk in batch_combined[k].keys():
                            batch_combined[k][kk] = torch.cat(
                                [batch_combined[k][kk], batch_per_dl[k][kk]], dim=0,
                            )
                    else:
                        raise NotImplementedError
        return batch_combined

    def compute_and_log_losses(
        self,
        losses: nn.ModuleList,
        prediction,
        batch: dict,
        gaussians,
        depth_dict: dict,
        global_step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Iterate over loss modules and accumulate total loss + per-loss values."""
        total_loss: Tensor = torch.tensor(0.0, device=self.device)
        loss_values: dict[str, float] = {}

        for loss_fn in losses:
            loss = loss_fn.forward(prediction, batch, gaussians, depth_dict, global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            try:
                loss_values[loss_fn.name] = float(loss.detach().item())
            except Exception:
                pass

            extra_logs = getattr(loss_fn, "extra_logs", None)
            if isinstance(extra_logs, dict):
                for k, v in extra_logs.items():
                    if torch.is_tensor(v):
                        self.log(f"loss/{k}", v)
                        try:
                            loss_values[k] = float(v.detach().item())
                        except Exception:
                            pass
            total_loss = total_loss + loss

        return total_loss, loss_values

    def finalize_training_step(
        self,
        total_loss: Tensor,
        loss_values: dict[str, float],
        batch: dict,
    ) -> Tensor:
        """Shared end-of-training-step: logging, nan/inf guard, step tracker, GC."""
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            try:
                loss_values["total"] = float(total_loss.detach().item())
            except Exception:
                pass
            keys = sorted(loss_values.keys())
            msg = ", ".join([f"{k}={loss_values[k]:.6g}" for k in keys])
            logger.info("loss breakdown: %s", msg)

        # Skip numerically broken batches.
        if not torch.isfinite(total_loss).all():
            logger.warning(
                "Skipping batch with non-finite loss (%s) at step %s on Rank %s",
                total_loss.detach() if hasattr(total_loss, "detach") else total_loss,
                self.global_step,
                self.global_rank,
            )
            return total_loss * 0.0

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            logger.info(
                "train step %s; scene = %s; context = %s; loss = %s",
                self.global_step,
                [x[:20] for x in batch["scene"]],
                batch["context"]["index"].tolist(),
                total_loss.item() if hasattr(total_loss, "item") else total_loss,
            )

        self.log("info/global_step", self.global_step)

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    def print_preview_metrics(
        self,
        metrics: dict[str, float | Tensor],
        methods: list[str] | None = None,
        overlap_tag: str | None = None,
    ) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metrics_sub_steps = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metrics_sub_steps[overlap_tag] = 1
            else:
                s = self.running_metrics_sub_steps[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {
                    k: ((s * v) + metrics[k]) / (s + 1)
                    for k, v in self.running_metrics_sub[overlap_tag].items()
                }
                self.running_metrics_sub_steps[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def _print_metrics(running_metric, methods=None):
            if methods is None:
                methods = ["ours"]
            table = []
            for method in methods:
                row = [
                    f"{running_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))
            headers = ["Method"] + metric_list
            logger.info("%s", tabulate(table, headers))

        logger.info("All Pairs:")
        _print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                logger.info("Overlap: %s", k)
                _print_metrics(v, methods)

    # ---- Optimizer ----

    def setup(self, stage: str) -> None:
        # requires_grad must reach its final state here: setup() runs before the
        # strategy wraps the module, while configure_optimizers runs after. The
        # DDP reducer only registers params that require grad at wrap time, so a
        # flip after wrapping either breaks or silently skips grad sync.
        freeze_kw = list(self.optimizer_cfg.freeze_keywords or [])
        if not freeze_kw:
            return
        hit_counts = dict.fromkeys(freeze_kw, 0)
        n_frozen = 0
        for name, param in self.named_parameters():
            matched = [kw for kw in freeze_kw if kw in name]
            for kw in matched:
                hit_counts[kw] += 1
            param.requires_grad = not matched
            n_frozen += bool(matched)
        missed = [kw for kw, n in hit_counts.items() if n == 0]
        if missed:
            raise ValueError(f"freeze_keywords matched no parameters: {missed}")
        if self.global_rank == 0:
            for kw, n in hit_counts.items():
                logger.info("[setup] freeze keyword %r -> %d params", kw, n)
            logger.info("[setup] frozen %d params total", n_frozen)

    def configure_optimizers(self):
        cfg = self.optimizer_cfg
        base_lr = cfg.lr

        param_groups_cfg = list(cfg.param_groups or [])
        if param_groups_cfg:
            zero_lr = [grp.keywords for grp in param_groups_cfg if grp.lr_multiplier <= 0]
            if zero_lr:
                raise ValueError(
                    f"param_groups with lr_multiplier <= 0: {zero_lr}; "
                    "use freeze_keywords to freeze params"
                )
            # Declarative: assign each param to first matching group
            group_params: list[list] = [[] for _ in range(len(param_groups_cfg) + 1)]
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                assigned = False
                for i, grp in enumerate(param_groups_cfg):
                    if any(kw in name for kw in grp.keywords):
                        group_params[i].append(param)
                        assigned = True
                        break
                if not assigned:
                    group_params[-1].append(param)  # default group

            param_dicts = []
            for i, grp in enumerate(param_groups_cfg):
                if group_params[i]:
                    param_dicts.append({
                        "params": group_params[i],
                        "lr": base_lr * grp.lr_multiplier,
                    })
            if group_params[-1]:
                param_dicts.append({
                    "params": group_params[-1],
                    "lr": base_lr * cfg.backbone_lr_multiplier,
                })

            if self.global_rank == 0:
                for i, grp in enumerate(param_groups_cfg):
                    logger.info("[configure_optimizers] param_groups[%d] keywords=%s lr_mult=%.2f -> %d params", i, grp.keywords, grp.lr_multiplier, len(group_params[i]))
                logger.info("[configure_optimizers] default (backbone_lr_mult=%.2f) -> %d params", cfg.backbone_lr_multiplier, len(group_params[-1]))
        else:
            # Legacy: new_param_keywords + backbone_lr_multiplier
            keywords = list(getattr(cfg, "new_param_keywords", []) or [])
            if not keywords:
                keywords = ["gaussian_param_head", "interm"]
            new_params, pretrained_params = [], []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if any(kw in name for kw in keywords):
                    new_params.append(param)
                else:
                    pretrained_params.append(param)

            if getattr(self, "global_rank", 0) == 0:
                logger.info(
                    "[configure_optimizers] new_param_keywords=%s; new=%d params, backbone=%d params",
                    keywords, len(new_params), len(pretrained_params),
                )

            param_dicts = [
                {"params": new_params, "lr": base_lr},
                {"params": pretrained_params, "lr": base_lr * cfg.backbone_lr_multiplier},
            ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=base_lr, weight_decay=0.05, betas=(0.9, 0.95),
        )
        # One multiplicative factor for all groups (linear warmup, then cosine
        # 1 -> 0.1) so every group keeps its configured lr ratio; a shared
        # absolute floor would flatten or invert the schedule of low-lr groups.
        warm_up_steps = cfg.warm_up_steps
        max_steps = get_cfg()["trainer"]["max_steps"]

        def lr_lambda(step: int) -> float:
            if step < warm_up_steps:
                return (step + 1) / warm_up_steps
            t = min(1.0, (step - warm_up_steps) / max(1, max_steps - warm_up_steps))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }

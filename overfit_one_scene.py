from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import wandb
from colorama import Fore
from hydra.core.hydra_config import HydraConfig
from jaxtyping import install_import_hook
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: Import `get_model` BEFORE install_import_hook().
# This matches `src/main.py` and prevents beartype/jaxtyping from type-checking
# HuggingFace `from_pretrained` config dicts against dataclass unions.
from src.model.model import get_model  # noqa: E402

import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint") 

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class EveryNStepsCheckpoint(Callback):
    """Save checkpoints every N optimizer steps (robust for overfit/debug runs).

    This avoids subtle interactions between Lightning's ModelCheckpoint logic and validation cadence
    when the training epoch has very few batches (often 1 in overfit settings).
    """

    def __init__(
        self,
        dirpath: Path,
        every_n_train_steps: int,
        save_weights_only: bool,
        keep_last_k: int,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.every_n_train_steps = int(every_n_train_steps)
        self.save_weights_only = bool(save_weights_only)
        self.keep_last_k = int(keep_last_k)

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero:
            return
        n = self.every_n_train_steps
        if n <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or (step % n) != 0:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"step={step}.ckpt"
        trainer.save_checkpoint(ckpt_path, weights_only=self.save_weights_only)

        # Optional retention to avoid filling disk during long overfit runs.
        if self.keep_last_k > 0:
            ckpts = sorted(self.dirpath.glob("step=*.ckpt"), key=lambda p: p.stat().st_mtime)
            if len(ckpts) > self.keep_last_k:
                for p in ckpts[: -self.keep_last_k]:
                    try:
                        p.unlink()
                    except Exception:
                        pass


class FixedOneSampleDataset(Dataset):
    """A finite dataset with exactly one sample (true overfit)."""

    def __init__(self, sample: dict[str, Any]) -> None:
        super().__init__()
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx != 0:
            raise IndexError(idx)
        return self.sample


@dataclass
class FixedViews:
    scene_id: str
    context: list[int]
    target: list[int]


class FixedViewSampler:
    """A view sampler that always returns the provided indices."""

    def __init__(self, context: list[int], target: list[int]) -> None:
        self._context = torch.tensor(context, dtype=torch.int64)
        self._target = torch.tensor(target, dtype=torch.int64)

    # Match DatasetCustom.getitem() call signature (it passes num_context_views).
    def sample(self, scene: str, num_context_views: int, extrinsics, intrinsics, device=torch.device("cpu")):
        return self._context.to(device), self._target.to(device), torch.tensor([0.5], device=device, dtype=torch.float32)

    @property
    def num_context_views(self) -> int:
        return int(self._context.numel())

    @property
    def num_target_views(self) -> int:
        return int(self._target.numel())


class FixedOverfitDataModule(LightningDataModule):
    def __init__(self, sample: dict[str, Any]) -> None:
        super().__init__()
        self._sample = sample

    def train_dataloader(self):
        # Cache loaders as attributes to match the original project's DataModule,
        # which ModelWrapper expects for epoch hooks.
        self.train_loader = DataLoader(
            FixedOneSampleDataset(self._sample),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return self.train_loader

    def val_dataloader(self):
        # Use the same fixed sample for visualization/validation.
        self.val_loader = DataLoader(
            FixedOneSampleDataset(self._sample),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return self.val_loader


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    # Configure beartype and jaxtyping.
    with install_import_hook(("src",), ("beartype", "beartype")):
        from src.config import load_typed_root_config
        from src.dataset.dataset_custom import DatasetCustom
        from src.global_cfg import set_cfg
        from src.loss import get_losses
        from src.misc.step_tracker import StepTracker
        from src.misc.wandb_tools import update_checkpoint_path
        from src.model.model_wrapper import ModelWrapper

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = Path(HydraConfig.get()["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(cyan(f"[overfit_one_scene] Saving outputs to {output_dir}"))

    # Logging
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        from src.misc.LocalLogger import LocalLogger

        logger = LocalLogger()

    # Checkpointing
    ckpt_dir = output_dir / "checkpoints"
    callbacks.append(
        EveryNStepsCheckpoint(
            ckpt_dir,
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_weights_only=cfg.checkpointing.save_weights_only,
            keep_last_k=cfg.checkpointing.save_top_k,
        )
    )
    # Always keep a deterministic "last" checkpoint, regardless of step trigger.
    callbacks.append(
        ModelCheckpoint(
            ckpt_dir,
            save_last=True,
            save_top_k=0,
            save_weights_only=cfg.checkpointing.save_weights_only,
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = "_"

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # Determinism (best-effort).
    torch.manual_seed(int(cfg_dict.seed))
    torch.cuda.manual_seed_all(int(cfg_dict.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Build model + wrapper.
    step_tracker = StepTracker(use_manager=False)
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        model,
        get_losses(cfg.loss),
        step_tracker,
    )

    # Build a fixed sample from the first dataset (expected: custom).
    if not cfg.dataset:
        raise ValueError("cfg.dataset is empty; expected a dataset config.")
    (dataset_wrapper,) = cfg.dataset[:1]
    # Typed wrapper is like DatasetCustomCfgWrapper(custom=...)
    dataset_field = next(iter(dataset_wrapper.__dict__.keys()))
    dataset_cfg = getattr(dataset_wrapper, dataset_field)

    # Instantiate dataset in train mode.
    # Note: DatasetCustom ignores augment/intr_augment unless those attrs exist.
    dummy_sampler = FixedViewSampler([0, 1], [2])
    ds = DatasetCustom(dataset_cfg, "train", dummy_sampler)  # type: ignore[arg-type]

    # Pick scene + fixed views (saved to output_dir for reuse).
    fixed_path = output_dir / "fixed_views.json"
    if fixed_path.exists():
        fixed_obj = json.loads(fixed_path.read_text())
        fixed = FixedViews(
            scene_id=str(fixed_obj["scene_id"]),
            context=[int(x) for x in fixed_obj["context"]],
            target=[int(x) for x in fixed_obj["target"]],
        )
        print(cyan(f"[overfit_one_scene] Loaded fixed views from {fixed_path}"))
    else:
        scene0 = ds.scenes[0]
        scene_id = str(scene0.get("scene_id", 0))
        frames = scene0.get("frames") or scene0.get("views") or scene0.get("images")
        if frames is None or len(frames) < 4:
            raise ValueError("Scene must have >= 4 frames to choose context/target views.")
        num_views = int(len(frames))

        # Choose 8 context views evenly spaced, and 2 targets in-between.
        num_ctx = min(8, num_views - 2)
        ctx = torch.linspace(0, num_views - 1, steps=num_ctx).round().to(torch.int64).tolist()
        ctx = sorted(set(int(x) for x in ctx))
        # Ensure we have at least 2 context views.
        while len(ctx) < 2:
            ctx.append(len(ctx))
        # Pick targets from remaining indices, roughly mid-range.
        remaining = [i for i in range(num_views) if i not in ctx]
        if len(remaining) < 1:
            remaining = [max(0, num_views // 2)]
        tgt = [remaining[len(remaining) // 3]]
        if len(remaining) > 1:
            tgt.append(remaining[(2 * len(remaining)) // 3])
        tgt = sorted(set(int(x) for x in tgt))
        fixed = FixedViews(scene_id=scene_id, context=ctx, target=tgt)
        fixed_path.write_text(json.dumps(fixed.__dict__, indent=2))
        print(cyan(f"[overfit_one_scene] Wrote fixed views to {fixed_path}"))

    # Swap in fixed sampler and materialize one deterministic sample.
    ds.view_sampler = FixedViewSampler(fixed.context, fixed.target)  # type: ignore[assignment]
    ps_h = int(dataset_cfg.input_image_shape[0] // 14)
    ps_w = int(dataset_cfg.input_image_shape[1] // 14)
    sample = ds.getitem(0, len(fixed.context), (ps_h, ps_w))
    print(
        cyan(
            f"[overfit_one_scene] scene_id={fixed.scene_id} "
            f"context={fixed.context} target={fixed.target} "
            f"patch={(ps_h*14, ps_w*14)}"
        )
    )

    datamodule = FixedOverfitDataModule(sample)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
        # Overfit runs often have 1 train batch/epoch; avoid log interval warnings.
        log_every_n_steps=1,
        # Skip sanity validation to make behavior deterministic for step-based checkpointing.
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        inference_mode=True,
    )

    trainer.fit(model_wrapper, datamodule=datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()


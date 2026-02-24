from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class DatasetCustomCfg(DatasetCfgCommon):
    name: Literal["custom"]
    root: Path
    manifest_path: Path
    depth_invalid_value: float = 1e9
    near: float = 0.01
    far: float = 100.0


@dataclass
class DatasetCustomCfgWrapper:
    custom: DatasetCustomCfg


class DatasetCustom(Dataset):
    """A manifest-driven dataset for custom multi-view scenes (e.g., infinigen)."""

    cfg: DatasetCustomCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor

    def __init__(self, cfg: DatasetCustomCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.root = cfg.root
        manifest = cfg.manifest_path
        if not manifest.is_absolute():
            manifest = self.root / manifest
        self.scenes = []
        if manifest.suffix.lower() == ".jsonl":
            with manifest.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.scenes.append(json.loads(line))
        else:
            with manifest.open("r") as f:
                obj = json.load(f)
            self.scenes = obj["scenes"] if isinstance(obj, dict) and "scenes" in obj else obj

        if cfg.overfit_to_scene is not None:
            self.scenes = [s for s in self.scenes if str(s.get("scene_id", "")) == cfg.overfit_to_scene]
            if not self.scenes:
                raise ValueError(f"overfit_to_scene={cfg.overfit_to_scene} not found in manifest")

        # Training requires enough views for the view sampler (>=2 and >= num_context_views
        # so that bounded/bounded_fixed samplers do not raise "Example does not have enough frames!").
        def _num_frames(s):
            for key in ("frames", "views", "images"):
                val = s.get(key)
                if val is not None and isinstance(val, list):
                    return len(val)
            return 0
        min_views = max(2, getattr(self.view_sampler, "num_context_views", 2))
        before = len(self.scenes)
        self.scenes = [s for s in self.scenes if _num_frames(s) >= min_views]
        if before > len(self.scenes):
            import warnings
            warnings.warn(
                f"DatasetCustom: dropped {before - len(self.scenes)} scene(s) with <{min_views} views (kept {len(self.scenes)})",
                UserWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.scenes)

    def _as_path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def _load_rgb(self, path: Path) -> Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_depth(self, path: Path) -> Tensor:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            arr = np.array(Image.open(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr)

    def _load_instance(self, path: Path) -> Tensor:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.int64)
        else:
            arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr.astype(np.int64))

    def _normalize_K(self, K_px: Tensor, h: int, w: int) -> Tensor:
        K = K_px.clone().to(torch.float32)
        K[0, :] /= float(w)
        K[1, :] /= float(h)
        return K

    def getitem(self, index: int, num_context_views: int, patchsize: tuple[int, int]) -> dict:
        scene = self.scenes[index]
        scene_id = str(scene.get("scene_id", index))
        if self.stage == "val":
            logger.info(
                "[DatasetCustom] Loading val sample scene_id=%s num_context_views=%s",
                scene_id,
                num_context_views,
            )
        frames = scene.get("frames") or scene.get("views") or scene.get("images")
        if frames is None or len(frames) < 2:
            raise ValueError("Scene must contain frames with >=2 views")

        # Build camera tensors for sampling.
        extrinsics = []
        intrinsics = []
        for fr in frames:
            c2w = torch.tensor(fr.get("c2w") or fr.get("extrinsic_c2w") or fr["camtoworld"], dtype=torch.float32)
            K_px = torch.tensor(fr.get("K") or fr.get("intrinsic") or fr["K_px"], dtype=torch.float32)
            hw = fr.get("HW")
            if hw is not None:
                h, w = int(hw[0]), int(hw[1])
            else:
                # fallback to config
                h, w = int(self.cfg.original_image_shape[0]), int(self.cfg.original_image_shape[1])
            intr = self._normalize_K(K_px, h, w)
            extrinsics.append(c2w)
            intrinsics.append(intr)
        extrinsics = torch.stack(extrinsics, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)

        context_indices, target_indices, overlap = self.view_sampler.sample(
            scene_id,
            num_context_views,
            extrinsics,
            intrinsics,
        )

        # Load selected frames.
        def load_stack(indices: Tensor):
            imgs, depths, insts, Ks = [], [], [], []
            for i in indices.tolist():
                fr = frames[int(i)]
                rgb_path = self._as_path(fr.get("rgb_path") or fr.get("image_path") or fr["rgb"])
                depth_path = self._as_path(fr.get("depth_path") or fr["depth"])
                inst_path = self._as_path(fr.get("instance_mask_path") or fr.get("mask_path") or fr["instance_mask"])

                img = self._load_rgb(rgb_path)
                depth = self._load_depth(depth_path)
                inst = self._load_instance(inst_path)

                imgs.append(img)
                depths.append(depth)
                insts.append(inst)

                # normalized K already built above, but keep aligned by index
                Ks.append(intrinsics[int(i)])

            images = torch.stack(imgs, dim=0)
            depth_t = torch.stack(depths, dim=0)
            inst_t = torch.stack(insts, dim=0)
            valid_mask = (depth_t > 0) & torch.isfinite(depth_t) & (depth_t < float(self.cfg.depth_invalid_value))
            return images, depth_t, inst_t, torch.stack(Ks, dim=0), valid_mask

        context_images, context_depths, context_inst, context_K, context_valid = load_stack(context_indices)
        target_images, target_depths, target_inst, target_K, target_valid = load_stack(target_indices)

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": context_K,
                "image": context_images,
                "depth": context_depths,
                "valid_mask": context_valid,
                "instance_mask": context_inst,
                "near": repeat(torch.tensor(self.cfg.near, dtype=torch.float32), " -> v", v=len(context_indices)),
                "far": repeat(torch.tensor(self.cfg.far, dtype=torch.float32), " -> v", v=len(context_indices)),
                "index": context_indices,
                "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": target_K,
                "image": target_images,
                "depth": target_depths,
                "valid_mask": target_valid,
                "instance_mask": target_inst,
                "near": repeat(torch.tensor(self.cfg.near, dtype=torch.float32), " -> v", v=len(target_indices)),
                "far": repeat(torch.tensor(self.cfg.far, dtype=torch.float32), " -> v", v=len(target_indices)),
                "index": target_indices,
                "overlap": overlap,
            },
            "scene": f"Custom {scene_id}",
        }

        # Crop to patchsize (same convention as other datasets).
        if self.stage == "train" and getattr(self.cfg, "intr_augment", False):
            intr_aug = True
        else:
            intr_aug = False
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)

        if self.stage == "train" and getattr(self.cfg, "augment", False):
            example = apply_augmentation_shim(example)

        return example

    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))


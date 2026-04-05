from __future__ import annotations

import json
import logging
import time
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
    """配置用于通用多视角 manifest 数据集的参数。

    该数据集通过 manifest 文件描述一组多视角场景，可用于：
    - InsScene 各子集（processed_infinigen / processed_scannetpp_v2 / processed_re10k 等）
    - 其他任意符合相同 schema 的多视角数据集

    manifest 中的每个 scene 应包含：
    - scene_id: 可选的场景 ID
    - frames / views / images: 视角列表，每个元素至少包含：
      - rgb_path / image_path / rgb: 相对 root 的 RGB 图像路径
      - depth_path / depth: 深度图路径（可选；例如 RE10K 可省略）
      - instance_mask_path / mask_path / instance_mask: 实例分割 mask 路径
      - K_px / K / intrinsic: 像素坐标系下的 3x3 内参矩阵
      - c2w / extrinsic_c2w / camtoworld: 4x4 相机位姿（camera-to-world）
      - HW: [H, W] 原始分辨率（可选，缺失时回退到 original_image_shape）
      - near / far: 可选的近平面/远平面（缺失时回退到本配置的 near/far）
    """

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
    """A manifest-driven dataset for custom multi-view scenes.

    Coordinate conventions (IMPORTANT):
        - ``c2w`` / ``extrinsic_c2w`` / ``camtoworld`` in the manifest **must** be
          a 4x4 **OpenCV camera-to-world** matrix:
              X → right,  Y → down,  Z → forward (looking direction).
        - ``K_px`` / ``K`` / ``intrinsic`` must be a 3x3 **pixel-unit** intrinsic
          matrix whose (cx, cy) follows **OpenCV convention** — the centre of the
          top-left pixel is at (0, 0).  If your intrinsics come from COLMAP
          (pixel centre at 0.5), subtract 0.5 from cx and cy before writing
          the manifest, or use ``src.coord.colmap_to_opencv_intrinsics``.
        - No coordinate-system conversion is applied at load time — the
          manifest must already be in OpenCV convention.
        - Output ``extrinsics``: **OpenCV c2w** (4x4).
        - Output ``intrinsics``: normalised K (fx,fy,cx,cy divided by W,H).
    """

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

    # Label string → integer mapping for physics labels.
    PHYS_LABEL_MAP = {"static": 1, "rigid": 2, "soft": 3, "unknown": 4}

    def _load_physics_labels(self, scene: dict) -> Tensor | None:
        """Load per-scene physics label JSON and return a 1-D lookup tensor.

        Returns a tensor of shape ``[max_instance_id + 1]`` where
        ``out[instance_id] = phys_class_int``.  Instance IDs without a label
        are mapped to 0 (ignore).  Returns ``None`` when the scene has no
        physics annotation.
        """
        plp = scene.get("physics_labels_path")
        if plp is None:
            return None
        path = self._as_path(plp)
        if not path.exists():
            logger.warning("[DatasetCustom] physics_labels_path %s not found, skipping", path)
            return None
        with path.open("r") as f:
            raw: dict[str, str] = json.load(f)
        if not raw:
            return None
        id_to_cls = {int(k): self.PHYS_LABEL_MAP.get(v, 0) for k, v in raw.items()}
        max_id = max(id_to_cls.keys())
        lut = torch.zeros(max_id + 1, dtype=torch.int64)
        for inst_id, cls_int in id_to_cls.items():
            lut[inst_id] = cls_int
        return lut

    def _normalize_K(self, K_px: Tensor, h: int, w: int) -> Tensor:
        K = K_px.clone().to(torch.float32)
        K[0, :] /= float(w)
        K[1, :] /= float(h)
        return K

    def getitem(self, index: int, num_context_views: int, patchsize: tuple[int, int]) -> dict:
        t0 = time.monotonic()
        scene = self.scenes[index]
        scene_id = str(scene.get("scene_id", index))
        if self.stage == "val":
            logger.info(
                "[DatasetCustom] Loading val sample scene_id=%s num_context_views=%s index=%s patch=%s",
                scene_id,
                num_context_views,
                index,
                patchsize,
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

        t_sample0 = time.monotonic()
        context_indices, target_indices, overlap = self.view_sampler.sample(
            scene_id,
            num_context_views,
            extrinsics,
            intrinsics,
        )
        t_sample = time.monotonic() - t_sample0

        # Check once whether this scene has depth (all frames consistent).
        scene_has_depth = any(
            (fr.get("depth_path") or fr.get("depth")) is not None for fr in frames
        )

        # Load selected frames.
        def load_stack(indices: Tensor):
            imgs, depths, insts, Ks = [], [], [], []
            for i in indices.tolist():
                fr = frames[int(i)]
                rgb_path = self._as_path(fr.get("rgb_path") or fr.get("image_path") or fr["rgb"])
                inst_path = self._as_path(fr.get("instance_mask_path") or fr.get("mask_path") or fr["instance_mask"])

                try:
                    img = self._load_rgb(rgb_path)
                    inst = self._load_instance(inst_path)

                    img_h, img_w = img.shape[-2], img.shape[-1]
                    if inst.shape[-2] != img_h or inst.shape[-1] != img_w:
                        inst = torch.nn.functional.interpolate(
                            inst.unsqueeze(0).unsqueeze(0).float(),
                            size=(img_h, img_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).long()

                    if scene_has_depth:
                        depth_raw = fr.get("depth_path") or fr.get("depth")
                        depth = self._load_depth(self._as_path(depth_raw))
                        if depth.shape[-2] != img_h or depth.shape[-1] != img_w:
                            depth = torch.nn.functional.interpolate(
                                depth.unsqueeze(0).unsqueeze(0),
                                size=(img_h, img_w),
                                mode="nearest",
                            ).squeeze(0).squeeze(0)
                    else:
                        depth = torch.ones(img_h, img_w, dtype=torch.float32)
                except Exception as e:
                    logger.exception(
                        "[DatasetCustom] Failed loading files for scene_id=%s frame_idx=%s rgb=%s inst=%s",
                        scene_id,
                        i,
                        str(rgb_path),
                        str(inst_path),
                    )
                    raise

                imgs.append(img)
                depths.append(depth)
                insts.append(inst)
                Ks.append(intrinsics[int(i)])

            images = torch.stack(imgs, dim=0)
            depth_t = torch.stack(depths, dim=0)
            inst_t = torch.stack(insts, dim=0)
            if scene_has_depth:
                valid_mask = (depth_t > 0) & torch.isfinite(depth_t) & (depth_t < float(self.cfg.depth_invalid_value))
            else:
                valid_mask = torch.ones_like(depth_t, dtype=torch.bool)
            return images, depth_t, inst_t, torch.stack(Ks, dim=0), valid_mask

        t_io0 = time.monotonic()
        context_images, context_depths, context_inst, context_K, context_valid = load_stack(context_indices)
        target_images, target_depths, target_inst, target_K, target_valid = load_stack(target_indices)
        t_io = time.monotonic() - t_io0

        phys_label_map = self._load_physics_labels(scene)

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

        if phys_label_map is not None:
            example["context"]["phys_label_map"] = phys_label_map
            example["target"]["phys_label_map"] = phys_label_map

        # Crop to patchsize (same convention as other datasets).
        if self.stage == "train" and getattr(self.cfg, "intr_augment", False):
            intr_aug = True
        else:
            intr_aug = False
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)

        if self.stage == "train" and getattr(self.cfg, "augment", False):
            example = apply_augmentation_shim(example)

        if self.stage == "val":
            # Only emit extra info when something is slow, to keep logs usable.
            dt = time.monotonic() - t0
            slow_s = float(getattr(self.cfg, "val_log_slow_threshold_s", 5.0)) if hasattr(self.cfg, "val_log_slow_threshold_s") else 5.0
            if dt >= slow_s:
                logger.warning(
                    "[DatasetCustom] Slow val sample scene_id=%s index=%s dt=%.3fs (sample=%.3fs io=%.3fs) ctx=%s tgt=%s",
                    scene_id,
                    index,
                    dt,
                    t_sample,
                    t_io,
                    context_indices.tolist(),
                    target_indices.tolist(),
                )

        return example

    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))


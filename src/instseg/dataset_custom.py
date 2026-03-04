from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .image_ops import (
    adjust_K_for_resize_crop_and_normalize,
    apply_resize_crop_depth,
    apply_resize_crop_instance_mask,
    apply_resize_crop_rgb,
    resize_to_cover_and_center_crop_params,
)
from .types import Stage
from .view_sampler import ViewSamplerCfg, sample_bounded


@dataclass
class DatasetCustomCfg:
    """通用多视角 manifest 数据集配置（InsScene 各子集 / Infinigen / ScanNet++ / RE10K 等）。

    manifest 文件可以是：
    - .jsonl：每行一个 scene（dict）
    - .json：list[scene] 或 {"scenes": [...]} 形式

    每个 scene 推荐包含：
    - scene_id: 可选场景 ID
    - frames / views / images: 视角列表（长度 >= 2），每个元素至少包含：
      - rgb_path / image_path / rgb: 相对 root 的 RGB 图像路径
      - depth_path / depth: 深度图路径（可选；无深度的数据集可省略该字段）
      - instance_mask_path / mask_path / instance_mask: 实例分割 mask 路径
      - K_px / K / intrinsic: 像素坐标系下的 3x3 内参矩阵
      - c2w / extrinsic_c2w / camtoworld: 4x4 相机位姿（camera-to-world）
      - near / far: 可选的近平面/远平面（缺失时使用本配置 near/far）
    """

    name: Literal["custom"]
    root: Path
    manifest_path: Path
    input_image_shape: list[int]  # [H, W]
    cameras_are_circular: bool = False
    overfit_to_scene: str | None = None
    view_sampler: ViewSamplerCfg | None = None

    # Instance mask semantics.
    instance_ignore_id: int = 0

    # Near/far fallbacks if not in manifest.
    near: float = 0.01
    far: float = 100.0

    # Depth invalid sentinel filter for all datasets (e.g. InsScene/Infinigen often use very large values like 1e10).
    depth_invalid_value: float = 1e9


@dataclass
class DatasetCustomCfgWrapper:
    custom: DatasetCustomCfg


def _as_path(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _load_json_any(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        scenes = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scenes.append(json.loads(line))
        return scenes
    with path.open("r") as f:
        return json.load(f)


def _load_rgb(path: Path) -> Tensor:
    img = Image.open(path).convert("RGB")
    return F.to_tensor(img)  # [3,H,W] in [0,1]


def _load_depth(path: Path) -> Tensor:
    # Supports .npy or image-like depth (png/exr). For image depths, user should ensure units.
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    return torch.from_numpy(arr)  # [H,W]


def _load_instance_mask(path: Path) -> Tensor:
    # Expect integer IDs stored as single-channel PNG/TIFF etc or .npy.
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # If RGB mask, take first channel (user should provide single-channel ideally).
        arr = arr[..., 0]
    return torch.from_numpy(arr.astype(np.int64))  # [H,W]


class DatasetCustom(Dataset):
    cfg: DatasetCustomCfg
    stage: Stage

    def __init__(self, cfg: DatasetCustomCfg, stage: Stage) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.root = cfg.root

        scenes = _load_json_any(_as_path(self.root, cfg.manifest_path))
        if isinstance(scenes, dict) and "scenes" in scenes:
            scenes = scenes["scenes"]
        if not isinstance(scenes, list):
            raise ValueError("manifest must be a list of scenes or {scenes: [...]}")

        # Optionally overfit.
        if cfg.overfit_to_scene is not None:
            scenes = [s for s in scenes if str(s.get("scene_id", "")) == cfg.overfit_to_scene]
            if not scenes:
                raise ValueError(f"overfit_to_scene={cfg.overfit_to_scene} not found in manifest")

        self.scenes = scenes
        if cfg.view_sampler is None:
            # Default bounded sampler similar to repo defaults.
            self.cfg.view_sampler = ViewSamplerCfg(  # type: ignore[call-arg]
                name="bounded",
                num_context_views=24,
                num_target_views=2,
                min_distance_between_context_views=8,
                max_distance_between_context_views=64,
                min_distance_to_context_views=0,
            )

    def __len__(self) -> int:
        return len(self.scenes)

    def _get_frame(self, scene: dict, idx: int) -> dict:
        frames = scene.get("frames") or scene.get("views") or scene.get("images")
        if frames is None:
            raise ValueError("Each scene must contain `frames` (list).")
        return frames[idx]

    def __getitem__(self, index: int) -> dict:
        scene = self.scenes[index]
        frames = scene.get("frames") or scene.get("views") or scene.get("images")
        if frames is None or not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("Scene must contain a list `frames` with >=2 entries.")

        num_views = len(frames)
        device = torch.device("cpu")
        ctx_idx, tgt_idx = sample_bounded(
            num_views,
            cfg=self.cfg.view_sampler,  # type: ignore[arg-type]
            device=device,
        )

        def load_stack(indices: Tensor):
            rgbs, depths, masks, c2ws, Ks, nears, fars = [], [], [], [], [], [], []
            for i in indices.tolist():
                fr = self._get_frame(scene, int(i))
                rgb_path = _as_path(self.root, fr.get("rgb_path") or fr.get("image_path") or fr["rgb"])
                depth_key = fr.get("depth_path") or fr.get("depth")
                depth_path = _as_path(self.root, depth_key) if depth_key is not None else None
                mask_path = _as_path(
                    self.root,
                    fr.get("instance_mask_path") or fr.get("mask_path") or fr["instance_mask"],
                )

                rgb = _load_rgb(rgb_path)
                depth = _load_depth(depth_path) if depth_path is not None else None
                inst = _load_instance_mask(mask_path)

                h_in, w_in = rgb.shape[-2:]
                h_out, w_out = self.cfg.input_image_shape
                p = resize_to_cover_and_center_crop_params(h_in, w_in, h_out, w_out)
                rgb = apply_resize_crop_rgb(rgb, p)
                depth = apply_resize_crop_depth(depth, p) if depth is not None else None
                inst = apply_resize_crop_instance_mask(inst, p)

                # Camera.
                c2w = torch.tensor(fr.get("c2w") or fr.get("extrinsic_c2w") or fr["camtoworld"], dtype=torch.float32)
                K = torch.tensor(fr.get("K") or fr.get("intrinsic") or fr["K_px"], dtype=torch.float32)
                K_norm = adjust_K_for_resize_crop_and_normalize(K, h_in, w_in, p)

                near = float(fr.get("near", self.cfg.near))
                far = float(fr.get("far", self.cfg.far))

                rgbs.append(rgb)
                depths.append(depth)
                masks.append(inst)
                c2ws.append(c2w)
                Ks.append(K_norm)
                nears.append(near)
                fars.append(far)

            images = torch.stack(rgbs, dim=0)  # [V,3,H,W] in [0,1]
            if any(d is None for d in depths):
                # 允许无深度数据集（例如 RE10K）；使用占位 0 深度和全 False valid_mask。
                if not all(d is None or isinstance(d, torch.Tensor) for d in depths):
                    raise ValueError("depths list must contain only Tensor or None")
                if any(d is None for d in depths) and not all(d is None for d in depths):
                    raise ValueError("Mixed depth presence per scene is not supported")
                h_out, w_out = self.cfg.input_image_shape
                depths_t = torch.zeros((len(depths), h_out, w_out), dtype=torch.float32)
                valid_mask = torch.zeros_like(depths_t, dtype=torch.bool)
            else:
                depths_t = torch.stack(depths, dim=0)  # [V,H,W]
                valid_mask = (depths_t > 0) & torch.isfinite(depths_t) & (
                    depths_t < float(self.cfg.depth_invalid_value)
                )  # [V,H,W]
            masks_t = torch.stack(masks, dim=0).to(torch.int64)  # [V,H,W]
            extr = torch.stack(c2ws, dim=0)  # [V,4,4]
            intr = torch.stack(Ks, dim=0)  # [V,3,3] normalized
            near_t = torch.tensor(nears, dtype=torch.float32)
            far_t = torch.tensor(fars, dtype=torch.float32)
            return images, depths_t, masks_t, extr, intr, near_t, far_t, valid_mask

        ctx = load_stack(ctx_idx)
        tgt = load_stack(tgt_idx)

        (ctx_img, ctx_depth, ctx_inst, ctx_extr, ctx_intr, ctx_near, ctx_far, ctx_valid) = ctx
        (tgt_img, tgt_depth, tgt_inst, tgt_extr, tgt_intr, tgt_near, tgt_far, tgt_valid) = tgt

        example = {
            "context": {
                "image": ctx_img,
                "depth": ctx_depth,
                "instance_mask": ctx_inst,
                "valid_mask": ctx_valid,
                "extrinsics": ctx_extr,
                "intrinsics": ctx_intr,
                "near": ctx_near,
                "far": ctx_far,
                "index": ctx_idx,
                "overlap": torch.zeros_like(ctx_near),
            },
            "target": {
                "image": tgt_img,
                "depth": tgt_depth,
                "instance_mask": tgt_inst,
                "valid_mask": tgt_valid,
                "extrinsics": tgt_extr,
                "intrinsics": tgt_intr,
                "near": tgt_near,
                "far": tgt_far,
                "index": tgt_idx,
                "overlap": torch.zeros_like(tgt_near),
            },
            "scene": [str(scene.get("scene_id", index))],
        }
        return example


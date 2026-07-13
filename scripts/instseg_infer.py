"""Unified AnySplat inference script with modular input/output stages.

Supports two input modes (custom images or dataset), multiple composable
output modes (rgb, gt, seg2d, pca2d, seg3d_ply, video, embedding, ply),
and two model loading paths (ModelWrapper+ckpt or HuggingFace pretrained).

Examples:
  # Custom images, default compare output:
  python scripts/anysplat_infer.py \
      --image_dir examples/vrnerf/riverview \
      --outputs seg2d,pca2d \
      --out_dir outputs/my_result

  # Randomly sample 8 views from image_dir (use --seed for reproducibility):
  python scripts/anysplat_infer.py \
      --image_dir examples/vrnerf/riverview \
      --max_views 8 \
      --outputs seg2d

  # Multiple clustering algorithms with overlay + compare (gt|rgb|seg):
  # (--ckpt optional, defaults to run_dir/checkpoints/last.ckpt)
  python scripts/anysplat_infer.py \
      --run_dir output/my_run \
      --outputs rgb,gt,seg2d \
      --cluster_algos kmeans,dbscan,hdbscan \
      --seg2d_outs seg,overlay,compare \
      --compare_style gt_rgb_seg

  # PCA with overlay and compare:
  python scripts/anysplat_infer.py \
      --image_dir examples/vrnerf/riverview \
      --outputs pca2d \
      --pca2d_outs pca,overlay,compare

  # Dataset mode: DEBUG logs for coordinate boundaries (logger ``coord``):
  python scripts/instseg_infer.py --run_dir output/my_run --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import os
import tempfile
from copy import deepcopy

import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coord import CameraConvention, ExtrinsicType, log_coordinate_op


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InferenceInput:
    """Unified container for model input, regardless of source."""
    images: Tensor                          # [1, V, 3, H, W] in [0, 1]
    meta: dict[str, Any] = field(default_factory=dict)
    gt_instance_mask: Tensor | None = None  # [1, V, H, W]
    depth: Tensor | None = None             # [1, V, H, W]
    intrinsics: Tensor | None = None        # [1, V, 3, 3]
    extrinsics: Tensor | None = None        # [1, V, 4, 4]


@dataclass
class OutputConfig:
    """Aggregated output-related CLI args."""
    k: int = 20
    cluster_algos: list[str] = field(default_factory=lambda: ["kmeans"])
    kmeans_iters: int = 30
    max_points: int = 200_000
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 10
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    hdbscan_cluster_selection_epsilon: float = 0.06
    opacity_threshold: float = 1.0 / 255.0
    seed: int = 0
    palette_seed: int = 0
    seg2d_outs: list[str] = field(default_factory=lambda: ["compare"])
    pca2d_outs: list[str] = field(default_factory=lambda: ["compare"])
    compare_style: str = "rgb_seg"
    save_unclustered: bool = False


# ---------------------------------------------------------------------------
# Input Stage
# ---------------------------------------------------------------------------

def _parse_int_list(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class _FixedSampler:
    def __init__(self, ctx: list[int], tgt: list[int]):
        self._ctx = torch.tensor(ctx, dtype=torch.int64)
        self._tgt = torch.tensor(tgt, dtype=torch.int64)

    @property
    def num_context_views(self) -> int:
        return int(self._ctx.numel())

    @property
    def num_target_views(self) -> int:
        return int(self._tgt.numel())

    def sample(self, scene, num_context_views, extrinsics, intrinsics, device=torch.device("cpu")):
        return self._ctx.to(device), self._tgt.to(device), torch.tensor([0.5], device=device, dtype=torch.float32)


class _RandomSampler:
    def __init__(self, num_ctx: int, num_tgt: int, seed: int):
        self._num_ctx = num_ctx
        self._num_tgt = num_tgt
        self._seed = seed

    @property
    def num_context_views(self) -> int:
        return self._num_ctx

    @property
    def num_target_views(self) -> int:
        return self._num_tgt

    def sample(self, scene, num_context_views, extrinsics, intrinsics, device=torch.device("cpu")):
        V = int(extrinsics.shape[0])
        need = self._num_ctx + self._num_tgt
        if V < need:
            raise ValueError(f"Scene has {V} views but need {need}")
        rng = random.Random(self._seed)
        idx = list(range(V))
        rng.shuffle(idx)
        ctx = torch.tensor(idx[: self._num_ctx], dtype=torch.int64, device=device)
        tgt = torch.tensor(idx[self._num_ctx : self._num_ctx + self._num_tgt], dtype=torch.int64, device=device)
        return ctx, tgt, torch.tensor([0.5], device=device, dtype=torch.float32)


def load_input_images(args: argparse.Namespace) -> InferenceInput:
    """Load input from a directory of custom images."""
    from src.utils.image import process_image

    image_dir = Path(args.image_dir)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    max_views = getattr(args, "max_views", None)
    if max_views is not None and max_views < len(paths):
        rng = random.Random(getattr(args, "seed", 0))
        paths = rng.sample(paths, max_views)
        paths = sorted(paths)  # keep deterministic order for reproducibility
        print(f"[input] Randomly sampled {max_views} views (seed={getattr(args, 'seed', 0)})")

    imgs = torch.stack([process_image(str(p)) for p in paths], dim=0)  # [V, 3, 448, 448] in [-1, 1]
    imgs_01 = (imgs + 1.0) * 0.5  # -> [0, 1]
    images = imgs_01.unsqueeze(0)  # [1, V, 3, H, W]

    meta = {
        "source": "images",
        "image_dir": str(image_dir),
        "image_paths": [str(p) for p in paths],
        "num_views": len(paths),
    }
    print(f"[input] Loaded {len(paths)} images from {image_dir}")
    return InferenceInput(images=images, meta=meta)


def load_input_video(args: argparse.Namespace) -> InferenceInput:
    """Load input by extracting frames from a video (e.g. 1 frame per second), then use image pipeline."""
    video_path = Path(args.input_video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps_interval = float(getattr(args, "video_fps", 1.0))  # frames per second to sample
    temp_dir = tempfile.mkdtemp(prefix="instseg_video_")
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps * fps_interval))
        count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                out_path = os.path.join(temp_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(out_path, frame)
                frame_idx += 1
        cap.release()
    except Exception:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if frame_idx == 0:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"No frames extracted from {video_path}")

    arg_copy = argparse.Namespace(
        image_dir=temp_dir,
        max_views=getattr(args, "max_views", None),
        seed=getattr(args, "seed", 0),
    )
    inp = load_input_images(arg_copy)
    inp.meta["source"] = "video"
    inp.meta["video_path"] = str(video_path)
    inp.meta["temp_dir"] = temp_dir
    inp.meta["num_frames_extracted"] = frame_idx
    print(f"[input] Extracted {frame_idx} frames from {video_path} (1 every {frame_interval} frames)")
    return inp


def _find_scene_index(scenes: list[dict[str, Any]], scene_id: str) -> int:
    for i, s in enumerate(scenes):
        if str(s.get("scene_id", i)) == str(scene_id):
            return i
    raise ValueError(f"scene_id={scene_id} not found ({len(scenes)} scenes)")


def load_input_dataset(args: argparse.Namespace) -> InferenceInput:
    """Load input from a dataset using the Hydra config in run_dir."""
    from omegaconf import OmegaConf

    from src.config import load_typed_root_config
    from src.dataset.dataset_manifest import DatasetManifest
    from src.global_cfg import set_cfg

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg_dict = OmegaConf.load(str(cfg_path))
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    manifest_cfg = None
    for w in cfg.dataset:
        if hasattr(w, "manifest"):
            manifest_cfg = w.manifest
            break
    if manifest_cfg is None:
        raise ValueError("No manifest dataset cfg found in cfg.dataset")

    cli_ctx = _parse_int_list(getattr(args, "context_views", None))
    cli_tgt = _parse_int_list(getattr(args, "target_views", None))

    if cli_ctx and cli_tgt:
        sampler = _FixedSampler(cli_ctx, cli_tgt)
    else:
        sampler = _RandomSampler(
            int(getattr(args, "num_context", 2)),
            int(getattr(args, "num_target", 1)),
            int(getattr(args, "seed", 0)),
        )

    ds = DatasetManifest(manifest_cfg, "train", sampler)

    scene_id_arg = getattr(args, "scene_id", None)
    if scene_id_arg:
        scene_index = _find_scene_index(ds.scenes, scene_id_arg)
        scene_id = scene_id_arg
    else:
        rng = np.random.default_rng(int(getattr(args, "seed", 0)))
        scene_index = int(rng.integers(0, len(ds.scenes)))
        scene_id = str(ds.scenes[scene_index].get("scene_id", scene_index))

    ps_h = int(manifest_cfg.input_image_shape[0] // 14)
    ps_w = int(manifest_cfg.input_image_shape[1] // 14)
    example = ds.getitem(scene_index, sampler.num_context_views, (ps_h, ps_w))

    ctx_img = example["context"]["image"].unsqueeze(0)   # [1, Vc, 3, H, W]
    tgt_img = example["target"]["image"].unsqueeze(0)     # [1, Vt, 3, H, W]
    images = torch.cat([ctx_img, tgt_img], dim=1)         # [1, V, 3, H, W]

    view_indices = example["context"]["index"].tolist() + example["target"]["index"].tolist()

    gt_mask = None
    if "instance_mask" in example["context"] and "instance_mask" in example["target"]:
        gt_mask = torch.cat([
            example["context"]["instance_mask"].unsqueeze(0),
            example["target"]["instance_mask"].unsqueeze(0),
        ], dim=1)

    depth = None
    if "depth" in example["context"] and "depth" in example["target"]:
        depth = torch.cat([
            example["context"]["depth"].unsqueeze(0),
            example["target"]["depth"].unsqueeze(0),
        ], dim=1)

    intrinsics = extrinsics = None
    if "intrinsics" in example["context"]:
        intrinsics = torch.cat([
            example["context"]["intrinsics"].unsqueeze(0),
            example["target"]["intrinsics"].unsqueeze(0),
        ], dim=1)
    if "extrinsics" in example["context"]:
        extrinsics = torch.cat([
            example["context"]["extrinsics"].unsqueeze(0),
            example["target"]["extrinsics"].unsqueeze(0),
        ], dim=1)

    meta = {
        "source": "dataset",
        "run_dir": str(run_dir),
        "scene_id": scene_id,
        "scene_index": scene_index,
        "context_views": example["context"]["index"].tolist(),
        "target_views": example["target"]["index"].tolist(),
        "view_indices": view_indices,
    }
    print(f"[input] Dataset scene_id={scene_id} context={meta['context_views']} target={meta['target_views']}")
    if extrinsics is not None:
        log_coordinate_op(
            "dataset_extrinsics",
            CameraConvention.OPENCV,
            ExtrinsicType.C2W,
            tuple(extrinsics.shape),
            context=f"scene_id={scene_id} (DatasetManifest → model)",
        )
    return InferenceInput(
        images=images,
        meta=meta,
        gt_instance_mask=gt_mask,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def load_input(args: argparse.Namespace) -> InferenceInput:
    if getattr(args, "input_video", None):
        return load_input_video(args)
    if getattr(args, "image_dir", None):
        return load_input_images(args)
    if getattr(args, "run_dir", None):
        return load_input_dataset(args)
    raise ValueError("Specify --input_video, --image_dir (images mode), or --run_dir (dataset mode)")


# ---------------------------------------------------------------------------
# Model Stage
# ---------------------------------------------------------------------------

def load_model_wrapper(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load model via Lightning wrapper + checkpoint (AnySplat or IGGT)."""
    from src.model.arch.weight_loading import load_model_from_run

    return load_model_from_run(args.run_dir, args.ckpt, device=device)


def load_model_pretrained(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load model via HF AnySplat, optionally with instance head."""
    from src.model.arch.anysplat import AnySplat
    from src.model.arch.weight_loading import init_anysplat_from_hf

    hf_id = getattr(args, "hf_model", "lhjiang/anysplat")
    instance_dim = int(getattr(args, "instance_feat_dim", 0))

    if instance_dim > 0:
        base = AnySplat.from_pretrained(hf_id)
        encoder_cfg = deepcopy(base.encoder_cfg)
        encoder_cfg.instance_feat_dim = instance_dim
        encoder_cfg.pretrained_weights = ""
        decoder_cfg = deepcopy(base.decoder_cfg)
        model = init_anysplat_from_hf(
            hf_id, encoder_cfg, decoder_cfg, base=base,
        )
        del base
        print(f"[model] HF model with instance_feat_dim={instance_dim}")
    else:
        model = AnySplat.from_pretrained(hf_id)
        print(f"[model] HF model {hf_id} (no instance head)")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if getattr(args, "run_dir", None) and getattr(args, "ckpt", None):
        return load_model_wrapper(args, device)
    else:
        return load_model_pretrained(args, device)


# ---------------------------------------------------------------------------
# Inference Stage
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, inp: InferenceInput, device: torch.device):
    """Run encoder forward pass. Returns EncoderOutput."""
    images = inp.images.to(device)
    return model.encoder(images, global_step=0, visualization_dump=None)


# ---------------------------------------------------------------------------
# Output: 2D Segmentation Maps
# ---------------------------------------------------------------------------

def _to_uint8_rgb(img_chw: Tensor) -> np.ndarray:
    return img_chw.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _save_image(path: Path, arr: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _blend_overlay(rgb: np.ndarray, vis: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend *vis* onto *rgb* where *vis* has non-zero pixels."""
    fg = (vis > 0).any(axis=-1, keepdims=True).astype(np.float32)
    return (rgb * (1.0 - alpha * fg) + vis * alpha * fg).clip(0, 255).astype(np.uint8)


def _algo_tag(cfg: OutputConfig, algo: str) -> str:
    if algo == "kmeans":
        return f"kmeans_k{cfg.k}"
    if algo == "dbscan":
        return f"dbscan_eps{cfg.dbscan_eps}_ms{cfg.dbscan_min_samples}"
    if algo == "hdbscan":
        return f"hdbscan_mcs{cfg.hdbscan_min_cluster_size}_ms{cfg.hdbscan_min_samples}"
    return algo


def _make_compare(
    style: str, rgb: np.ndarray, vis_col: np.ndarray,
    gt_col: np.ndarray | None,
) -> np.ndarray | None:
    """Horizontally concatenate images based on *compare_style*."""
    if style == "rgb_seg":
        return np.concatenate([rgb, vis_col], axis=1)
    if style == "gt_seg":
        return np.concatenate([gt_col, vis_col], axis=1) if gt_col is not None else None
    if style == "gt_rgb_seg":
        return np.concatenate([gt_col, rgb, vis_col], axis=1) if gt_col is not None else None
    return None


def _colorize_gt_masks(gt_mask: Tensor, palette_seed: int) -> list[np.ndarray]:
    """Colorize GT instance masks. Returns list of [H, W, 3] uint8 arrays."""
    from src.visualization.instance_viz import colorize_labels, make_color_lut

    gt_np = gt_mask[0].detach().cpu().numpy().astype(np.int32)  # [V, H, W]
    unique_ids = np.unique(gt_np)
    unique_ids = unique_ids[unique_ids != 0]
    gt_lut = make_color_lut(max(1, len(unique_ids) + 1), seed=palette_seed + 1)
    id_map = {int(uid): j + 1 for j, uid in enumerate(unique_ids.tolist())}
    gt_contig = np.zeros_like(gt_np, dtype=np.int32)
    for uid, j in id_map.items():
        gt_contig[gt_np == uid] = j
    V = gt_np.shape[0]
    return [colorize_labels(gt_contig[vi], gt_lut, ignore_label=0) for vi in range(V)]


def _cluster_2d(
    feat: Tensor, valid: Tensor | None, algo: str, cfg: OutputConfig,
) -> np.ndarray:
    """Cluster 2D instance feature maps. Returns labels [V, H, W] int32 numpy."""
    from src.visualization.instance_viz import cluster_instance_embeddings

    if algo == "kmeans":
        # Helper returns 0 for invalid/background and 1..K for valid clusters.
        return cluster_instance_embeddings(
            feat, valid, k=cfg.k, max_points=cfg.max_points,
            num_iters=cfg.kmeans_iters, seed=cfg.seed,
        )

    V, D, H, W = feat.shape
    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, D)  # [V*H*W, D]
    if valid is not None:
        valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    else:
        valid_idx = torch.arange(feat_flat.shape[0], device=feat.device)

    x = F.normalize(feat_flat[valid_idx].float(), p=2, dim=-1, eps=1e-8).cpu().numpy()

    S = min(cfg.max_points, len(x))
    if S < len(x):
        rng = np.random.default_rng(cfg.seed)
        sub_idx = rng.choice(len(x), S, replace=False)
        x_fit = x[sub_idx]
    else:
        x_fit = x

    if algo == "hdbscan":
        from src.instseg.hdbscan_assign import hdbscan_assign
        all_labels = hdbscan_assign(
            x,
            cluster_selection_epsilon=cfg.hdbscan_cluster_selection_epsilon,
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            max_points=cfg.max_points,
            rng_seed=cfg.seed,
        )
        labels_flat = np.zeros(V * H * W, dtype=np.int32)
        labels_flat[valid_idx.cpu().numpy()] = all_labels + 1
        return labels_flat.reshape(V, H, W)

    if algo == "dbscan":
        from sklearn.cluster import DBSCAN
        cl = DBSCAN(
            eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples,
            metric="cosine", n_jobs=-1,
        ).fit_predict(x_fit)
    else:
        raise ValueError(f"Unknown cluster algo: {algo}")

    is_cluster = cl >= 0
    n_clusters = int(cl[is_cluster].max() + 1) if np.any(is_cluster) else 0
    centers = np.zeros((n_clusters, D), dtype=np.float32)
    for cid in range(n_clusters):
        c = x_fit[cl == cid].mean(0)
        centers[cid] = c / (np.linalg.norm(c) + 1e-8)

    labels_flat = np.zeros(V * H * W, dtype=np.int32)
    if n_clusters > 0:
        sims = x @ centers.T
        labels_flat[valid_idx.cpu().numpy()] = sims.argmax(axis=1).astype(np.int32) + 1
    return labels_flat.reshape(V, H, W)


# ---------------------------------------------------------------------------
# Output: RGB images (top-level, deduplicated)
# ---------------------------------------------------------------------------

def output_rgb(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Save per-view RGB images to out_dir/rgb/."""
    V = inp.images.shape[1]
    view_indices = inp.meta.get("view_indices", list(range(V)))
    rgb_dir = out_dir / "rgb"
    for vi in range(V):
        idx = view_indices[vi] if vi < len(view_indices) else vi
        _save_image(rgb_dir / f"{idx:04d}.png", _to_uint8_rgb(inp.images[0, vi]))
    print(f"[rgb] Saved {V} views to {rgb_dir}")


# ---------------------------------------------------------------------------
# Output: GT instance masks (top-level, deduplicated)
# ---------------------------------------------------------------------------

def output_gt(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Save per-view colorized GT instance masks to out_dir/gt/."""
    if inp.gt_instance_mask is None:
        print("[gt] Skipped: no gt_instance_mask")
        return
    V = inp.images.shape[1]
    view_indices = inp.meta.get("view_indices", list(range(V)))
    gt_cols = _colorize_gt_masks(inp.gt_instance_mask, cfg.palette_seed)
    gt_dir = out_dir / "gt"
    for vi in range(V):
        idx = view_indices[vi] if vi < len(view_indices) else vi
        _save_image(gt_dir / f"{idx:04d}.png", gt_cols[vi])
    print(f"[gt] Saved {V} views to {gt_dir}")


# ---------------------------------------------------------------------------
# Output: 2D Segmentation Maps
# ---------------------------------------------------------------------------

def output_seg2d(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Cluster instance_feat_map with one or more algorithms and save selectively."""
    from src.visualization.instance_viz import colorize_labels, make_color_lut

    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        print("[seg2d] Skipped: instance_feat_map is None (need instance_feat_dim > 0)")
        return

    feat = feat_map[0].float()  # [V, D, H, W]
    V = feat.shape[0]

    valid = None
    if inp.gt_instance_mask is not None:
        valid = (inp.gt_instance_mask[0] > 0).to(feat.device)
    elif hasattr(enc_out, "valid_mask") and enc_out.valid_mask is not None:
        valid = enc_out.valid_mask[0].to(torch.bool).to(feat.device)

    view_indices = inp.meta.get("view_indices", list(range(V)))
    images = inp.images[0]  # [V, 3, H, W]
    outs = set(cfg.seg2d_outs)

    compare_style = cfg.compare_style
    if compare_style in ("gt_seg", "gt_rgb_seg") and inp.gt_instance_mask is None:
        print(f"[seg2d] No GT available, falling back compare_style from '{compare_style}' to 'rgb_seg'")
        compare_style = "rgb_seg"

    need_rgb = outs & {"overlay", "compare"}
    need_gt = ("compare" in outs
               and compare_style in ("gt_seg", "gt_rgb_seg")
               and inp.gt_instance_mask is not None)

    gt_cols: list[np.ndarray] | None = None
    if need_gt:
        gt_cols = _colorize_gt_masks(inp.gt_instance_mask, cfg.palette_seed)

    for algo in cfg.cluster_algos:
        labels = _cluster_2d(feat, valid, algo, cfg)
        tag = _algo_tag(cfg, algo)
        algo_dir = out_dir / "seg2d" / tag
        max_label = int(labels.max()) if labels.size > 0 else 0
        lut = make_color_lut(max(1, max_label + 1), seed=cfg.palette_seed)

        for vi in range(V):
            idx = view_indices[vi] if vi < len(view_indices) else vi
            seg_col = colorize_labels(labels[vi], lut, ignore_label=0)
            rgb = _to_uint8_rgb(images[vi]) if need_rgb else None
            gt_col = gt_cols[vi] if gt_cols is not None else None

            if "seg" in outs:
                _save_image(algo_dir / "seg" / f"{idx:04d}.png", seg_col)
            if "overlay" in outs:
                _save_image(algo_dir / "overlay" / f"{idx:04d}.png",
                            _blend_overlay(rgb, seg_col))
            if "compare" in outs:
                cmp = _make_compare(compare_style, rgb, seg_col, gt_col)
                if cmp is not None:
                    _save_image(algo_dir / "compare" / f"{idx:04d}.png", cmp)

        algo_dir.mkdir(parents=True, exist_ok=True)
        np.save(algo_dir / "labels.npy", labels)
        print(f"[seg2d] algo={tag} saved {V} views (outs={cfg.seg2d_outs}) to {algo_dir}")


# ---------------------------------------------------------------------------
# Output: 2D PCA Visualization
# ---------------------------------------------------------------------------

def output_pca2d(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """PCA-reduce instance_feat_map to 3-channel RGB and save selectively."""
    from src.visualization.instance_viz import pca_visualize_embeddings

    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        print("[pca2d] Skipped: instance_feat_map is None")
        return

    feat = feat_map[0].float()  # [V, D, H, W]
    V = feat.shape[0]

    valid = None
    if inp.gt_instance_mask is not None:
        valid = (inp.gt_instance_mask[0] > 0).to(feat.device)

    pca_vis = pca_visualize_embeddings(feat, valid)  # [V, 3, H, W] in [0, 1]

    pca_dir = out_dir / "pca2d"
    view_indices = inp.meta.get("view_indices", list(range(V)))
    images = inp.images[0]
    outs = set(cfg.pca2d_outs)
    need_rgb = outs & {"overlay", "compare"}

    for vi in range(V):
        idx = view_indices[vi] if vi < len(view_indices) else vi
        pca_rgb = _to_uint8_rgb(pca_vis[vi])
        rgb = _to_uint8_rgb(images[vi]) if need_rgb else None

        if "pca" in outs:
            _save_image(pca_dir / "pca" / f"{idx:04d}.png", pca_rgb)
        if "overlay" in outs:
            _save_image(pca_dir / "overlay" / f"{idx:04d}.png",
                        _blend_overlay(rgb, pca_rgb))
        if "compare" in outs:
            _save_image(pca_dir / "compare" / f"{idx:04d}.png",
                        np.concatenate([rgb, pca_rgb], axis=1))

    print(f"[pca2d] Saved {V} views (outs={cfg.pca2d_outs}) to {pca_dir}")


# ---------------------------------------------------------------------------
# Output: 3D Cluster PLY
# ---------------------------------------------------------------------------

@torch.no_grad()
def _cluster_3d_kmeans(
    feat: Tensor, opacities: Tensor, k: int, iters: int,
    seed: int, max_points: int, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    from src.instseg.kmeans import kmeans_torch
    G, N = feat.shape
    valid = opacities.reshape(-1) > threshold
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return np.zeros(G, dtype=np.int32), np.zeros((k, N), dtype=np.float32)

    S = min(max_points, valid_idx.numel())
    g = torch.Generator(device=feat.device)
    g.manual_seed(seed)
    perm = torch.randperm(valid_idx.numel(), generator=g, device=feat.device)[:S]
    x = F.normalize(feat[valid_idx[perm]].float(), p=2, dim=-1, eps=1e-8)
    _, centers = kmeans_torch(x, k=k, num_iters=iters, seed=seed)

    feat_n = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)
    centers_n = F.normalize(centers.float(), p=2, dim=-1, eps=1e-8)
    pred0 = (feat_n @ centers_n.T).argmax(dim=1).to(torch.int64)
    pred0[~valid] = -1

    labels = torch.zeros(G, dtype=torch.int32, device=feat.device)
    labels[pred0 >= 0] = (pred0[pred0 >= 0] + 1).to(torch.int32)
    return labels.cpu().numpy(), centers.cpu().numpy()


@torch.no_grad()
def _cluster_3d_dbscan(
    feat: Tensor, opacities: Tensor,
    eps: float, min_samples: int, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import DBSCAN
    G, N = feat.shape
    valid = opacities.reshape(-1) > threshold
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return np.zeros(G, dtype=np.int32), np.zeros((0, N), dtype=np.float32)

    x_np = F.normalize(feat[valid_idx].float(), p=2, dim=-1, eps=1e-8).cpu().numpy()
    db_labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit_predict(x_np)

    labels = np.zeros(G, dtype=np.int32)
    valid_np = valid_idx.cpu().numpy()
    is_cluster = db_labels >= 0
    if np.any(is_cluster):
        labels[valid_np[is_cluster]] = (db_labels[is_cluster] + 1).astype(np.int32)

    n_clusters = int(db_labels[is_cluster].max() + 1) if np.any(is_cluster) else 0
    centers = np.zeros((n_clusters, N), dtype=np.float32)
    for cid in range(n_clusters):
        mem = valid_np[db_labels == cid]
        c = feat[torch.from_numpy(mem).to(feat.device)].float().mean(0)
        centers[cid] = F.normalize(c, p=2, dim=-1, eps=1e-8).cpu().numpy()
    return labels, centers


@torch.no_grad()
def _cluster_3d_hdbscan(
    feat: Tensor, opacities: Tensor,
    min_cluster_size: int, min_samples: int, threshold: float,
    cluster_selection_epsilon: float = 0.06, max_points: int = 200_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    from src.instseg.hdbscan_assign import hdbscan_assign

    G, N = feat.shape
    valid = opacities.reshape(-1) > threshold
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return np.zeros(G, dtype=np.int32), np.zeros((0, N), dtype=np.float32)

    x_np = F.normalize(feat[valid_idx].float(), p=2, dim=-1, eps=1e-8).cpu().numpy()
    valid_labels = hdbscan_assign(
        x_np,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_points=max_points,
        rng_seed=seed,
    )

    labels = np.zeros(G, dtype=np.int32)
    valid_np = valid_idx.cpu().numpy()
    labels[valid_np] = valid_labels + 1

    n_clusters = int(valid_labels.max()) + 1
    centers = np.zeros((n_clusters, N), dtype=np.float32)
    for cid in range(n_clusters):
        mem = valid_np[valid_labels == cid]
        c = feat[torch.from_numpy(mem).to(feat.device)].float().mean(0)
        centers[cid] = F.normalize(c, p=2, dim=-1, eps=1e-8).cpu().numpy()
    return labels, centers


def _run_cluster_3d(
    feat: Tensor, op: Tensor, algo: str, cfg: OutputConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if algo == "kmeans":
        return _cluster_3d_kmeans(
            feat, op, k=cfg.k, iters=cfg.kmeans_iters, seed=cfg.seed,
            max_points=cfg.max_points, threshold=cfg.opacity_threshold,
        )
    if algo == "dbscan":
        return _cluster_3d_dbscan(
            feat, op, eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples, threshold=cfg.opacity_threshold,
        )
    if algo == "hdbscan":
        return _cluster_3d_hdbscan(
            feat, op, min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples, threshold=cfg.opacity_threshold,
            cluster_selection_epsilon=cfg.hdbscan_cluster_selection_epsilon,
            max_points=cfg.max_points, seed=cfg.seed,
        )
    raise ValueError(f"Unknown cluster_algo: {algo}")


def output_seg3d_ply(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Cluster gaussian_instance_feat, encode as SH colors, export PLY."""
    from src.model.ply_export import export_ply
    from src.utils.spherical_harmonics import rgb_to_sh
    from src.visualization.instance_viz import make_color_lut

    gaussians = enc_out.gaussians
    g_feat = enc_out.gaussian_instance_feat
    if g_feat is None:
        print("[seg3d_ply] Skipped: gaussian_instance_feat is None")
        return

    device = gaussians.means.device
    feat = g_feat[0].to(device).float()  # [G, N]
    op = gaussians.opacities[0].to(device)
    pos_np = gaussians.means[0].detach().cpu().numpy().astype(np.float32)

    for algo in cfg.cluster_algos:
        labels, centers = _run_cluster_3d(feat, op, algo, cfg)
        tag = _algo_tag(cfg, algo)

        max_label = int(labels.max()) if labels.size > 0 else 0
        lut = make_color_lut(max_label + 1, seed=cfg.palette_seed)
        rgb_u8 = lut[np.clip(labels, 0, max_label)]  # [G, 3]
        rgb_f = torch.from_numpy(rgb_u8).to(device=device, dtype=torch.float32) / 255.0

        sh_dc = rgb_to_sh(rgb_f)
        harm = torch.zeros_like(gaussians.harmonics[0]).float()
        harm[:, :, 0] = sh_dc

        ply_dir = out_dir / "seg3d_ply" / tag
        ply_dir.mkdir(parents=True, exist_ok=True)

        export_ply(
            gaussians.means[0], gaussians.scales[0], gaussians.rotations[0],
            harm, gaussians.opacities[0], ply_dir / "gaussians_cluster_color.ply",
            save_sh_dc_only=True,
        )

        from plyfile import PlyData, PlyElement
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        verts = np.empty(len(pos_np), dtype=dtype)
        verts["x"], verts["y"], verts["z"] = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
        verts["red"], verts["green"], verts["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
        PlyData([PlyElement.describe(verts, "vertex")]).write(
            str(ply_dir / "colored_pointcloud.ply"))

        np.save(ply_dir / "cluster_labels.npy", labels)
        n_valid = int((labels > 0).sum())
        print(f"[seg3d_ply] algo={tag} clusters={max_label} "
              f"valid={n_valid}/{labels.size} -> {ply_dir}")


# ---------------------------------------------------------------------------
# Output: Split 3DGS into per-cluster PLY files
# ---------------------------------------------------------------------------

def output_seg3d_split(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Cluster gaussian_instance_feat, then export each cluster as an independent 3DGS PLY."""
    from src.model.ply_export import export_ply

    gaussians = enc_out.gaussians
    g_feat = enc_out.gaussian_instance_feat
    if g_feat is None:
        print("[seg3d_split] Skipped: gaussian_instance_feat is None")
        return

    device = gaussians.means.device
    feat = g_feat[0].to(device).float()  # [G, N]
    op = gaussians.opacities[0].to(device)

    for algo in cfg.cluster_algos:
        labels, _ = _run_cluster_3d(feat, op, algo, cfg)
        tag = _algo_tag(cfg, algo)
        split_dir = out_dir / "seg3d_split" / tag
        split_dir.mkdir(parents=True, exist_ok=True)

        labels_t = torch.from_numpy(labels).to(device)
        max_label = int(labels.max()) if labels.size > 0 else 0
        n_saved = 0

        for cid in range(1, max_label + 1):
            mask = labels_t == cid
            if mask.sum() == 0:
                continue
            export_ply(
                gaussians.means[0][mask],
                gaussians.scales[0][mask],
                gaussians.rotations[0][mask],
                gaussians.harmonics[0][mask],
                gaussians.opacities[0][mask],
                split_dir / f"cluster_{cid - 1:03d}.ply",
                save_sh_dc_only=True,
            )
            n_saved += 1

        if cfg.save_unclustered:
            unc_mask = labels_t == 0
            if unc_mask.sum() > 0:
                export_ply(
                    gaussians.means[0][unc_mask],
                    gaussians.scales[0][unc_mask],
                    gaussians.rotations[0][unc_mask],
                    gaussians.harmonics[0][unc_mask],
                    gaussians.opacities[0][unc_mask],
                    split_dir / "unclustered.ply",
                    save_sh_dc_only=True,
                )

        np.save(split_dir / "cluster_labels.npy", labels)
        print(f"[seg3d_split] algo={tag} saved {n_saved} cluster PLYs "
              f"(max_label={max_label}) -> {split_dir}")


# ---------------------------------------------------------------------------
# Output: Interpolated RGB/Depth Video
# ---------------------------------------------------------------------------

def output_video(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path, *, model=None) -> None:
    """Render interpolated RGB and depth videos from predicted poses."""
    from src.misc.image_io import save_interpolated_video

    pred_pose = enc_out.pred_context_pose
    if pred_pose is None:
        print("[video] Skipped: pred_context_pose is None")
        return
    if model is None:
        print("[video] Skipped: model reference needed for decoder")
        return

    gaussians = enc_out.gaussians
    _, _, _, h, w = inp.images.shape

    video_dir = out_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    save_interpolated_video(
        pred_pose["extrinsic"], pred_pose["intrinsic"],
        1, h, w, gaussians, str(video_dir), model.decoder,
    )
    print(f"[video] Saved rgb.mp4 and depth.mp4 to {video_dir}")


# ---------------------------------------------------------------------------
# Output: Raw Embedding Export
# ---------------------------------------------------------------------------

def output_embedding(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Export raw gaussian instance embedding as a torch checkpoint."""
    from src.instseg.export import export_gaussian_instance_embedding

    if enc_out.gaussian_instance_feat is None:
        print("[embedding] Skipped: gaussian_instance_feat is None")
        return

    emb_dir = out_dir / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = export_gaussian_instance_embedding(
        emb_dir / "gaussian_instance_embedding.pt",
        enc_out.gaussians,
        enc_out.gaussian_instance_feat,
        meta=inp.meta,
    )
    print(f"[embedding] Saved to {path}")


# ---------------------------------------------------------------------------
# Output: Plain PLY (original colors)
# ---------------------------------------------------------------------------

def output_ply(enc_out, inp: InferenceInput, cfg: OutputConfig, out_dir: Path) -> None:
    """Export vanilla Gaussian Splatting PLY with original SH colors."""
    from src.model.ply_export import export_ply

    gaussians = enc_out.gaussians
    ply_path = out_dir / "gaussians.ply"
    export_ply(
        gaussians.means[0], gaussians.scales[0], gaussians.rotations[0],
        gaussians.harmonics[0], gaussians.opacities[0], ply_path,
        save_sh_dc_only=True,
    )
    print(f"[ply] Saved to {ply_path}")


# ---------------------------------------------------------------------------
# Output Registry
# ---------------------------------------------------------------------------

OUTPUT_REGISTRY: dict[str, Callable] = {
    "rgb": output_rgb,
    "gt": output_gt,
    "seg2d": output_seg2d,
    "pca2d": output_pca2d,
    "seg3d_ply": output_seg3d_ply,
    "seg3d_split": output_seg3d_split,
    "video": output_video,
    "embedding": output_embedding,
    "ply": output_ply,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified AnySplat inference with composable outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g_input = ap.add_argument_group("Input (choose one)")
    g_input.add_argument("--input_video", type=str, default=None, help="Input video path; frames extracted (e.g. 1/sec) then used as images")
    g_input.add_argument("--video_fps", type=float, default=1.0, help="Sample 1 frame every N seconds from video (default: 1.0)")
    g_input.add_argument("--image_dir", type=str, default=None, help="Directory of input images (images mode)")
    g_input.add_argument("--max_views", type=int, default=None, help="Randomly sample N views from image_dir (default: use all)")
    g_input.add_argument("--run_dir", type=str, default=None, help="Hydra run dir with .hydra/config.yaml (dataset mode)")
    g_input.add_argument("--scene_id", type=str, default=None, help="Scene ID in the dataset (dataset mode)")
    g_input.add_argument("--context_views", type=str, default=None, help="Comma-separated context view indices")
    g_input.add_argument("--target_views", type=str, default=None, help="Comma-separated target view indices")
    g_input.add_argument("--num_context", type=int, default=2, help="Random context views count (dataset mode)")
    g_input.add_argument("--num_target", type=int, default=1, help="Random target views count (dataset mode)")

    g_model = ap.add_argument_group("Model")
    g_model.add_argument("--ckpt", type=str, default=None, help="Lightning .ckpt path (default: run_dir/checkpoints/last.ckpt when --run_dir is set)")
    g_model.add_argument("--hf_model", type=str, default="lhjiang/anysplat", help="HuggingFace model ID")
    g_model.add_argument("--instance_feat_dim", type=int, default=16, help="Instance feature dim (0 to disable)")
    g_model.add_argument("--device", type=str, default="cuda")

    g_output = ap.add_argument_group("Output")
    g_output.add_argument(
        "--outputs", type=str, default="seg2d,pca2d,seg3d_ply",
        help=f"Comma-separated output modes: {','.join(OUTPUT_REGISTRY.keys())}",
    )
    g_output.add_argument("--out_dir", type=str, default="outputs/anysplat_infer")

    g_2d = ap.add_argument_group("2D output control")
    g_2d.add_argument(
        "--seg2d_outs", type=str, default="compare",
        help="Comma-separated sub-outputs for seg2d: seg,overlay,compare (default: compare)",
    )
    g_2d.add_argument(
        "--pca2d_outs", type=str, default="compare",
        help="Comma-separated sub-outputs for pca2d: pca,overlay,compare (default: compare)",
    )
    g_2d.add_argument(
        "--compare_style", type=str, default="rgb_seg",
        choices=["rgb_seg", "gt_seg", "gt_rgb_seg"],
        help="Compare layout for seg2d (default: rgb_seg)",
    )

    g_cluster = ap.add_argument_group("Clustering parameters")
    g_cluster.add_argument(
        "--cluster_algos", type=str, default="kmeans",
        help="Comma-separated clustering algorithms: kmeans,dbscan,hdbscan (default: kmeans)",
    )
    g_cluster.add_argument("--k", type=int, default=20, help="K-means clusters")
    g_cluster.add_argument("--kmeans_iters", type=int, default=30)
    g_cluster.add_argument("--max_points", type=int, default=200_000, help="Max points for all clustering algorithms")
    g_cluster.add_argument("--dbscan_eps", type=float, default=0.3)
    g_cluster.add_argument("--dbscan_min_samples", type=int, default=10)
    g_cluster.add_argument("--hdbscan_min_cluster_size", type=int, default=50)
    g_cluster.add_argument("--hdbscan_min_samples", type=int, default=10)
    g_cluster.add_argument("--hdbscan_cluster_selection_epsilon", type=float, default=0.06,
                           help="HDBSCAN cluster_selection_epsilon "
                                "(aligned with iggt_idmap.py --eps, default: 0.06)")
    g_cluster.add_argument("--opacity_threshold", type=float, default=1.0 / 255.0)
    g_cluster.add_argument("--seed", type=int, default=0)
    g_cluster.add_argument("--palette_seed", type=int, default=0)
    g_cluster.add_argument(
        "--save_unclustered", action="store_true", default=False,
        help="Also save unclustered (label=0) Gaussians as unclustered.ply in seg3d_split",
    )

    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging for the coord logger (coordinate-system boundaries)",
    )

    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.verbose:
        _coord_log = logging.getLogger("coord")
        _coord_log.setLevel(logging.DEBUG)
        if not _coord_log.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setLevel(logging.DEBUG)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _coord_log.addHandler(_h)
        _coord_log.propagate = False

    # Auto-resolve ckpt when run_dir is set but ckpt is not (only if default path exists)
    if getattr(args, "run_dir", None) and getattr(args, "ckpt", None) is None:
        default_ckpt = Path(args.run_dir) / "checkpoints" / "last.ckpt"
        if default_ckpt.exists():
            args.ckpt = str(default_ckpt)
        # else: leave ckpt=None, load_model will use HF pretrained (run_dir may be for dataset only)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [s.strip() for s in args.outputs.split(",") if s.strip()]
    unknown = [r for r in requested if r not in OUTPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown output modes: {unknown}. Available: {list(OUTPUT_REGISTRY.keys())}")

    # --- Load input ---
    inp = load_input(args)

    # --- Load model ---
    model = load_model(args, device)

    # --- Inference ---
    print("[infer] Running encoder forward ...")
    enc_out = run_inference(model, inp, device)
    gs_info = f"Gaussians: {tuple(enc_out.gaussians.means.shape)}" if enc_out.gaussians is not None else "Gaussians: None (encoder-only)"
    print(f"[infer] {gs_info}, "
          f"instance_feat_map={'yes' if enc_out.instance_feat_map is not None else 'no'}, "
          f"gaussian_instance_feat={'yes' if enc_out.gaussian_instance_feat is not None else 'no'}")

    # --- Optional 3D KNN smoothing (aligns with iggt_idmap / base_wrapper) ---
    if enc_out.instance_feat_map is not None:
        depth_dict = enc_out.depth_dict or {}
        if depth_dict.get("world_points") is not None:
            from src.visualization.instance_viz import knn_smooth_instance_features
            feat0 = enc_out.instance_feat_map[0].float()
            feat0_smooth = knn_smooth_instance_features(feat0, depth_dict, k=20)
            enc_out.instance_feat_map = feat0_smooth.unsqueeze(0).to(
                dtype=enc_out.instance_feat_map.dtype
            )
            print("[knn] Applied 3D KNN smoothing (k=20) to instance_feat_map")
        else:
            print("[knn] Skipped: world_points not available in depth_dict")

    # --- Run outputs ---
    cluster_algos = [s.strip() for s in args.cluster_algos.split(",") if s.strip()]
    for a in cluster_algos:
        if a not in ("kmeans", "dbscan", "hdbscan"):
            raise ValueError(f"Unknown cluster algo: {a}. Choose from: kmeans, dbscan, hdbscan")
    seg2d_outs = [s.strip() for s in args.seg2d_outs.split(",") if s.strip()]
    pca2d_outs = [s.strip() for s in args.pca2d_outs.split(",") if s.strip()]

    out_cfg = OutputConfig(
        k=args.k, cluster_algos=cluster_algos, kmeans_iters=args.kmeans_iters,
        max_points=args.max_points, dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        opacity_threshold=args.opacity_threshold, seed=args.seed,
        palette_seed=args.palette_seed,
        seg2d_outs=seg2d_outs, pca2d_outs=pca2d_outs,
        compare_style=args.compare_style,
        save_unclustered=args.save_unclustered,
    )

    for name in requested:
        fn = OUTPUT_REGISTRY[name]
        print(f"\n--- Output: {name} ---")
        if name == "video":
            fn(enc_out, inp, out_cfg, out_dir, model=model)
        else:
            fn(enc_out, inp, out_cfg, out_dir)

    # --- Save meta ---
    meta = {**inp.meta, "outputs": requested, "args": vars(args)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n[done] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()

"""
IGGT Multi-View Feature Clustering ID Map
==========================================

Uses IGGTModel (lightweight, no detectron2/torch_geometric deps) to
extract per-pixel part features from multi-view images, then performs
cross-view HDBSCAN clustering to produce consistent ID maps.

Usage:
    python iggt_idmap.py \
        --image_dir <directory containing multi-view images> \
        --model_path <path to IGGT checkpoint> \
        --output_dir output_idmap \
        --image_size 504,336 \
        --hdbscan_min_cluster_size 500 \
        --hdbscan_min_samples 100 \
        --eps 0.06
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Model loading -- uses the new IGGTModel from src/model/arch/iggt.py
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str = "cuda"):
    """Load the IGGT model for part-feature extraction using the integrated module."""
    from src.model.encoder.iggt import EncoderIGGTCfg
    from src.model.arch.iggt import IGGTModel

    logger.info("Loading IGGT model from %s ...", model_path)
    cfg = EncoderIGGTCfg(
        name="iggt",
        instance_feat_dim=8,
        freeze_backbone=True,
        pretrained_weights="",
    )
    model = IGGTModel.from_checkpoint(cfg, model_path, device=device)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    logger.info("Model loaded (feat_dim=8)")
    return model


# ---------------------------------------------------------------------------
# Image loading (lightweight, no IGGT/ deps)
# ---------------------------------------------------------------------------

def discover_images(image_dir: str) -> list[str]:
    """Find all image files in *image_dir* (non-recursive), sorted by name."""
    paths = []
    for p in sorted(os.listdir(image_dir)):
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(image_dir, p))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    logger.info("Found %d images in %s", len(paths), image_dir)
    return paths


def load_and_preprocess_images_resize(image_path_list, resize_target_size):
    """Resize all input images to target (W, H), output [V, 3, H, W]."""
    import torchvision.transforms as T

    target_w, target_h = int(resize_target_size[0]), int(resize_target_size[1])
    to_tensor = T.ToTensor()
    images = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))
    return torch.stack(images, dim=0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, image_paths, image_size, device="cuda", batch_size=4):
    """Run IGGT on all images and return (part_feat [V, D, H, W], pts3d [V, 3, H, W])."""
    all_feats = []
    all_pts3d = []
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_paths = image_paths[start:end]
        images = load_and_preprocess_images_resize(batch_paths, image_size).to(device)
        images_batch = images.unsqueeze(0)  # [1, V_batch, 3, H, W]
        enc_out, _ = model(images_batch)
        part_feat = enc_out.instance_feat_map[0].float()  # [V_batch, D, H, W]
        pts3d = enc_out.depth_dict["world_points"][0].float()  # [V_batch, H, W, 3]
        pts3d = pts3d.permute(0, 3, 1, 2)  # [V_batch, 3, H, W]
        for vi in range(part_feat.shape[0]):
            all_feats.append(part_feat[vi].cpu())
            all_pts3d.append(pts3d[vi].cpu())
        logger.info("  Processed views %d-%d / %d", start, end, len(image_paths))

    return torch.stack(all_feats, dim=0), torch.stack(all_pts3d, dim=0)


# ---------------------------------------------------------------------------
# 3D KNN feature smoothing (lightweight scipy replacement for torch_geometric)
# ---------------------------------------------------------------------------

def knn_smooth_features(
    pts3d: torch.Tensor,
    feat: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    """Smooth instance features by averaging k nearest neighbours in 3D space.

    Replaces ``knn_avg_features_pyg`` from the original IGGT pipeline without
    requiring torch_geometric / torch_scatter.

    Args:
        pts3d: [V, 3, H, W] world coordinates.
        feat:  [V, D, H, W] instance features (should already be L2-normalised).
        k:     Number of 3D nearest neighbours.

    Returns:
        Smoothed features [V, D, H, W].
    """
    from scipy.spatial import cKDTree

    V, D, H, W = feat.shape
    N = V * H * W

    points_np = pts3d.permute(0, 2, 3, 1).reshape(N, 3).numpy()   # (N, 3)
    feat_np = feat.permute(0, 2, 3, 1).reshape(N, D).numpy()       # (N, D)

    logger.info("Building cKDTree for %d points (k=%d) ...", N, k)
    tree = cKDTree(points_np)
    _, indices = tree.query(points_np, k=k + 1)  # +1 because self is included
    indices = indices[:, 1:]                       # exclude self

    logger.info("Averaging neighbour features ...")
    smoothed = feat_np[indices].mean(axis=1)       # (N, D)

    smoothed_t = torch.from_numpy(smoothed).reshape(V, H, W, D).permute(0, 3, 1, 2)
    return smoothed_t


# ---------------------------------------------------------------------------
# Clustering (HDBSCAN, no torch_geometric / torch_scatter)
# ---------------------------------------------------------------------------

def cluster_features_hdbscan(
    feat_vdhw: torch.Tensor,
    eps: float = 0.06,
    min_samples: int = 100,
    min_cluster_size: int = 500,
    max_cluster_points: int = 200_000,
) -> np.ndarray:
    """Subsample -> HDBSCAN -> NearestCentroid assign all pixels.

    Expects features to be already normalised by the caller.

    Args:
        feat_vdhw: [V, D, H, W] feature tensor (pre-normalised).
        max_cluster_points: Max pixels fed into HDBSCAN (default 200k).

    Returns:
        id_maps: [V, H, W] int32, contiguous IDs starting from 0.
    """
    from src.instseg.hdbscan_assign import hdbscan_assign

    V, D, H, W = feat_vdhw.shape
    all_pixels = feat_vdhw.permute(0, 2, 3, 1).contiguous().reshape(-1, D).float().numpy()

    labels = hdbscan_assign(
        all_pixels,
        cluster_selection_epsilon=eps,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_points=max_cluster_points,
        rng_seed=0,
    )
    return labels.reshape(V, H, W)


# ---------------------------------------------------------------------------
# PCA visualization (from src/visualization/instance_viz.py, lightweight)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pca_visualize(feat_vdhw: torch.Tensor, low_p=0.02, high_p=0.98) -> np.ndarray:
    """PCA-reduce [V, D, H, W] features to [V, H, W, 3] uint8 RGB."""
    V, D, H, W = feat_vdhw.shape
    feat_flat = feat_vdhw.permute(0, 2, 3, 1).reshape(-1, D).float()
    feat_flat = F.normalize(feat_flat, dim=-1)

    mean = feat_flat.mean(dim=0, keepdim=True)
    feat_centered = feat_flat - mean

    _, _, v = torch.pca_lowrank(feat_centered, q=min(D, 256))
    proj = feat_centered @ v[:, :3]

    # Subsample for quantile when tensor is large (torch.quantile fails on >2^31 elements)
    max_quantile_elems = 1_000_000
    for i in range(3):
        ch = proj[:, i].flatten()
        n = ch.numel()
        if n > max_quantile_elems:
            step = max(1, n // max_quantile_elems)
            ch_sub = ch[::step]  # deterministic stride sampling
        else:
            ch_sub = ch
        v_lo = torch.quantile(ch_sub, low_p)
        v_hi = torch.quantile(ch_sub, high_p)
        if v_hi > v_lo:
            proj[:, i] = (proj[:, i] - v_lo) / (v_hi - v_lo)
        else:
            proj[:, i] = 0.5
    proj = proj.clamp(0, 1)

    return (proj.view(V, H, W, 3).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lut = rng.integers(32, 255, size=(num_colors, 3), dtype=np.uint8)
    return lut


def save_results(
    id_maps: np.ndarray,
    pca_rgb: np.ndarray,
    image_paths: list[str],
    output_dir: str,
):
    """Save ID maps (.npy), coloured ID map images, and PCA images."""
    idmap_dir = os.path.join(output_dir, "id_maps")
    vis_dir = os.path.join(output_dir, "id_maps_vis")
    pca_dir = os.path.join(output_dir, "pca_vis")
    os.makedirs(idmap_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)

    num_ids = int(id_maps.max()) + 1
    lut = make_color_lut(num_ids)

    V = id_maps.shape[0]
    for i in range(V):
        stem = os.path.splitext(os.path.basename(image_paths[i]))[0]

        np.save(os.path.join(idmap_dir, f"{stem}.npy"), id_maps[i])

        colored = lut[id_maps[i]]
        Image.fromarray(colored).save(os.path.join(vis_dir, f"{stem}.png"))

        Image.fromarray(pca_rgb[i]).save(os.path.join(pca_dir, f"{stem}.png"))

    np.save(os.path.join(output_dir, "id_maps.npy"), id_maps)
    logger.info("Saved %d ID maps (%d unique IDs) to %s", V, num_ids, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="IGGT multi-view feature clustering ID map")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing multi-view images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to IGGT model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_idmap",
                        help="Output directory")
    parser.add_argument("--image_size", type=str, default="504,336",
                        help="Resize target as W,H (default: 504,336)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Views per forward pass")
    parser.add_argument("--eps", type=float, default=0.06,
                        help="HDBSCAN cluster_selection_epsilon")
    parser.add_argument("--hdbscan_min_samples", type=int, default=100,
                        help="HDBSCAN min_samples")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=500,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--max_cluster_points", type=int, default=200_000,
                        help="Max pixels subsampled for HDBSCAN (default: 200000)")
    parser.add_argument("--knn_k", type=int, default=20,
                        help="K for 3D KNN feature smoothing (0 to disable)")
    parser.add_argument("--spatial_weight", type=float, default=0.0,
                        help="Weight for appending normalised xyz to clustering features "
                             "(0 = disabled, try 0.1~0.5)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    iggt_w, iggt_h = [int(x) for x in args.image_size.split(",")]
    image_size = (iggt_w, iggt_h)

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = discover_images(args.image_dir)
    model = load_model(args.model_path, args.device)

    logger.info("Running inference (image_size=%dx%d) ...", iggt_w, iggt_h)
    feat, pts3d = run_inference(model, image_paths, image_size,
                                device=args.device, batch_size=args.batch_size)
    # feat: [V, D, H, W], pts3d: [V, 3, H, W]
    del model
    torch.cuda.empty_cache()

    # L2 normalise part features
    V, D, H, W = feat.shape
    feat = F.normalize(feat.float(), dim=1)

    # 3D KNN feature smoothing
    if args.knn_k > 0:
        logger.info("3D KNN smoothing (k=%d) ...", args.knn_k)
        feat = knn_smooth_features(pts3d, feat, k=args.knn_k)

    # PCA on smoothed features (before spatial concat)
    logger.info("Computing PCA visualization ...")
    pca_rgb = pca_visualize(feat)

    # Optionally append normalised spatial coordinates for clustering
    cluster_feat = feat
    if args.spatial_weight > 0:
        logger.info("Appending spatial coords (weight=%.3f) ...", args.spatial_weight)
        xyz = pts3d.float()                                  # [V, 3, H, W]
        xyz_flat = xyz.permute(1, 0, 2, 3).reshape(3, -1)   # [3, V*H*W]
        xyz_mean = xyz_flat.mean(dim=1)                      # [3]
        xyz_std = xyz_flat.std(dim=1).clamp(min=1e-6)        # [3]
        xyz_norm = (xyz - xyz_mean.view(1, 3, 1, 1)) / xyz_std.view(1, 3, 1, 1)
        xyz_norm = xyz_norm * args.spatial_weight
        cluster_feat = torch.cat([feat, xyz_norm], dim=1)    # [V, D+3, H, W]

    logger.info("Clustering features to ID map ...")
    id_maps = cluster_features_hdbscan(
        cluster_feat,
        eps=args.eps,
        min_samples=args.hdbscan_min_samples,
        min_cluster_size=args.hdbscan_min_cluster_size,
        max_cluster_points=args.max_cluster_points,
    )

    save_results(id_maps, pca_rgb, image_paths, args.output_dir)
    logger.info("Done in %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()

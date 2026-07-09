"""Instance embedding visualization utilities: clustering and PCA.

Reusable helpers for visualizing instance_feat_map from instance heads.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def knn_smooth_instance_features(
    feat_map: Tensor,
    depth_dict: dict,
    k: int = 20,
) -> Tensor:
    """Smooth instance features by averaging k nearest neighbours in 3D space.

    Aligns with iggt_idmap.py pipeline for consistent cross-view instance IDs.
    Requires depth_dict["world_points"] (e.g. from IGGT encoder).

    Args:
        feat_map: [V, N, H, W] instance embeddings.
        depth_dict: Dict with "world_points" [B, V, H', W', 3] (batch 0 used).
        k: Number of 3D nearest neighbours.

    Returns:
        Smoothed feat_map [V, N, H, W], or original if world_points unavailable.
    """
    world_pts = depth_dict.get("world_points")
    if world_pts is None:
        return feat_map

    try:
        # world_points: [B, V, H', W', 3] or [V, H', W', 3] -> [V, 3, H', W']
        pts3d = world_pts.float()
        if pts3d.dim() == 5:
            pts3d = pts3d[0]  # [V, H', W', 3]
        pts3d = pts3d.permute(0, 3, 1, 2)  # [V, 3, H', W']

        V, N, H, W = feat_map.shape
        if pts3d.shape[2] != H or pts3d.shape[3] != W:
            pts3d = F.interpolate(
                pts3d,
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )

        feat = F.normalize(feat_map.float(), p=2, dim=1, eps=1e-8)
        smoothed = _knn_smooth_features_impl(pts3d.cpu(), feat.cpu(), k=k)
        return smoothed.to(device=feat_map.device, dtype=feat_map.dtype)
    except Exception as e:
        logger.warning("knn_smooth_instance_features failed (%s), using raw features", e)
        return feat_map


def _knn_smooth_features_impl(
    pts3d: Tensor,
    feat: Tensor,
    k: int = 20,
) -> Tensor:
    """Core KNN smoothing using scipy cKDTree (no torch_geometric)."""
    from scipy.spatial import cKDTree

    V, D, H, W = feat.shape
    N = V * H * W

    points_np = pts3d.permute(0, 2, 3, 1).reshape(N, 3).numpy()
    feat_np = feat.permute(0, 2, 3, 1).reshape(N, D).numpy()

    tree = cKDTree(points_np)
    _, indices = tree.query(points_np, k=k + 1)
    indices = indices[:, 1:]

    smoothed = feat_np[indices].mean(axis=1)
    return torch.from_numpy(smoothed).reshape(V, H, W, D).permute(0, 3, 1, 2)


def make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    """Create a deterministic color lookup table for instance IDs."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    lut = np.maximum(lut, 32).astype(np.uint8)
    return lut


def colorize_labels(
    labels_hw: np.ndarray,
    lut: np.ndarray,
    ignore_label: int = 0,
) -> np.ndarray:
    """Colorize integer label map using a lookup table.

    Args:
        labels_hw: [H, W] integer labels.
        lut: [num_colors, 3] uint8 RGB colors.
        ignore_label: Pixels with this label stay black.

    Returns:
        [H, W, 3] uint8 RGB image.
    """
    h, w = labels_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    mask = labels_hw != ignore_label
    if mask.any():
        idx = np.clip(labels_hw[mask], 0, lut.shape[0] - 1)
        out[mask] = lut[idx]
    return out


@torch.no_grad()
def cluster_instance_embeddings(
    feat_vnhw: Tensor,
    valid_vhw: Tensor | None = None,
    k: int = 20,
    max_points: int = 50000,
    num_iters: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Cluster per-pixel instance embeddings via k-means.

    Args:
        feat_vnhw: [V, N, H, W] instance embeddings.
        valid_vhw: [V, H, W] bool mask; if None, all pixels are valid.
        k: Number of clusters.
        max_points: Max pixels to sample for k-means (for speed).
        num_iters: K-means iterations.
        seed: Random seed.

    Returns:
        [V, H, W] int32 cluster labels.
        Label 0 is reserved for invalid/background; valid clusters are 1..K.
    """
    from src.instseg.kmeans import kmeans_torch

    V, N, H, W = feat_vnhw.shape
    feat = feat_vnhw.permute(0, 2, 3, 1).reshape(-1, N)
    valid = valid_vhw.reshape(-1) if valid_vhw is not None else torch.ones(V * H * W, dtype=torch.bool, device=feat.device)

    idx_all = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        return np.zeros((V, H, W), dtype=np.int32)

    S = min(int(max_points), int(idx_all.numel()))
    g = torch.Generator(device=feat.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(idx_all.numel(), generator=g, device=feat.device)[:S]
    idx = idx_all[perm]
    x = feat[idx]
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    _, centers = kmeans_torch(x, k=k, num_iters=num_iters, seed=seed)
    feat_n = F.normalize(feat, p=2, dim=-1, eps=1e-8)
    dot = feat_n @ centers.T
    pred = (2.0 - 2.0 * dot).argmin(dim=1).to(torch.int64)
    labels = torch.zeros(V * H * W, dtype=torch.int32, device=feat.device)
    labels[valid] = (pred[valid] + 1).to(torch.int32)
    return labels.view(V, H, W).cpu().numpy().astype(np.int32)


def pca_visualize_embeddings(
    feat_vnhw: Tensor,
    valid_vhw: Tensor | None = None,
    low_p: float = 0.02,
    high_p: float = 0.98,
) -> Tensor:
    """Reduce instance embeddings to 3 channels via PCA for RGB visualization.

    Aligns with iggt_idmap.py: L2 normalize first, then center (no std division).

    Args:
        feat_vnhw: [V, N, H, W] instance embeddings.
        valid_vhw: [V, H, W] bool mask; if None, all pixels are valid.
        low_p: Lower percentile for normalization.
        high_p: Upper percentile for normalization.

    Returns:
        [V, 3, H, W] float tensor in [0, 1] for RGB visualization.
    """
    MIN_SAMPLES_FOR_PCA = 10

    V, N, H, W = feat_vnhw.shape
    feat_flat = feat_vnhw.permute(0, 2, 3, 1).reshape(-1, N).float()
    feat_flat = F.normalize(feat_flat, p=2, dim=-1, eps=1e-8)

    if valid_vhw is not None:
        valid_flat = valid_vhw.reshape(-1)
        feat_valid = feat_flat[valid_flat]
        n_valid = feat_valid.shape[0]
        if n_valid == 0:
            logging.warning(
                "pca_visualize_embeddings: valid mask 下无有效像素，改用全图做 PCA"
            )
            feat_valid = feat_flat
        elif n_valid < MIN_SAMPLES_FOR_PCA:
            logging.warning(
                "pca_visualize_embeddings: 有效像素过少 (%d)，改用全图做 PCA",
                n_valid,
            )
            feat_valid = feat_flat
    else:
        feat_valid = feat_flat

    C = feat_valid.shape[1]
    if C < 3:
        pad_valid = torch.zeros((feat_valid.shape[0], 3 - C), device=feat_valid.device, dtype=feat_valid.dtype)
        feat_valid = torch.cat([feat_valid, pad_valid], dim=1)
        pad_flat = torch.zeros((feat_flat.shape[0], 3 - C), device=feat_flat.device, dtype=feat_flat.dtype)
        feat_flat = torch.cat([feat_flat, pad_flat], dim=1)
        C = 3

    # Align with iggt_idmap: center only (no std). PCA on L2-normalized features.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        feat_valid = feat_valid.float()
        feat_flat = feat_flat.float()
        mean = feat_valid.mean(dim=0, keepdim=True)
        feat_centered_valid = feat_valid - mean
        feat_centered_flat = feat_flat - mean
        try:
            _, _, v = torch.pca_lowrank(feat_centered_valid, q=min(C, 256))
            proj = torch.matmul(feat_centered_flat, v[:, :3])
        except Exception as e:
            logging.warning(
                "pca_visualize_embeddings: PCA 失败 (%s)，改用全图重试",
                e,
                exc_info=True,
            )
            try:
                mean = feat_flat.mean(dim=0, keepdim=True)
                feat_centered_flat = feat_flat - mean
                _, _, v = torch.pca_lowrank(feat_centered_flat, q=min(C, 256))
                proj = torch.matmul(feat_centered_flat, v[:, :3])
            except Exception as e2:
                logging.warning(
                    "pca_visualize_embeddings: 全图 PCA 仍失败，返回全黑 (%s)",
                    e2,
                    exc_info=True,
                )
                return torch.zeros((V, 3, H, W), device=feat_vnhw.device, dtype=feat_vnhw.dtype)

    # Percentile normalization: use valid pixels only for range so content is visible.
    # Stride-subsample before torch.quantile to avoid slowness / OOM on very large tensors,
    # aligned with iggt_idmap.pca_visualize.
    if valid_vhw is not None:
        valid_flat = valid_vhw.reshape(-1)
        if valid_flat.any():
            ch_sample = proj[valid_flat]
        else:
            ch_sample = proj
    else:
        ch_sample = proj
    max_quantile_elems = 1_000_000
    for i in range(3):
        ch = proj[:, i]
        ch_s = ch_sample[:, i].flatten()
        n = ch_s.numel()
        if n > max_quantile_elems:
            step = max(1, n // max_quantile_elems)
            ch_s = ch_s[::step]
        v_low = torch.quantile(ch_s, low_p)
        v_high = torch.quantile(ch_s, high_p)
        if v_high > v_low:
            proj[:, i] = (ch - v_low) / (v_high - v_low)
        else:
            proj[:, i] = 0.5
    proj = proj.clamp(0, 1)

    out = proj.view(V, H, W, 3).permute(0, 3, 1, 2)
    # Do not mask by valid_vhw for visualization - show PCA for all pixels
    return out

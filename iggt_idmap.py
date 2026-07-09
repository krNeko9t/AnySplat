"""
IGGT Multi-View Feature Clustering ID Map
==========================================

Uses IGGTModel (integrated in-repo, no Detectron2) to extract per-pixel part
features from multi-view images, then cross-view HDBSCAN clustering for ID maps.

Dependencies:
    - Inference: torchvision, ``src.model.arch.iggt``.
    - Optional 3D KNN smoothing (--knn_k > 0): ``torch_geometric`` + ``torch_scatter``
      for ``--knn_backend pyg``, or SciPy ``cKDTree`` for ``--knn_backend scipy`` / ``auto``.
    - Very large ``V*H*W`` (e.g. hundreds of views): use ``--max_views``, ``--knn_backend scipy``,
      ``--knn_query_chunk`` (SciPy chunked query, global scope), or ``--knn_scope per_view``.

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
import random
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

# Above this many points, ``auto`` backend skips PyG global kNN (GPU OOM risk) and uses SciPy + chunk.
PYG_KNN_POINT_THRESHOLD = 5_000_000
# When ``auto`` forces SciPy and ``knn_query_chunk`` is 0, use this chunk size to cap peak RAM.
KNN_QUERY_CHUNK_AUTO_FALLBACK = 262_144


def _proc_memory_mb() -> float | None:
    """Best-effort current process RSS in MiB (Linux: /proc/self/statm)."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * page_size / (1024.0 * 1024.0)
    except (OSError, AttributeError, ValueError, IndexError):
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux: kilobytes; macOS/BSD: bytes — detect heuristically
            if ru > 10**9:  # absurd as KB → assume bytes
                return ru / (1024.0 * 1024.0)
            if ru > 10**7:  # likely bytes
                return ru / (1024.0 * 1024.0)
            return ru / 1024.0  # KiB → MiB on Linux
        except Exception:
            return None


def _mem_suffix() -> str:
    m = _proc_memory_mb()
    return f"RSS~{m:.1f}MiB" if m is not None else "RSS=n/a"


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


def subsample_image_paths(image_paths: list[str], max_views: int | None, seed: int) -> list[str]:
    """Randomly subsample paths; return sorted paths for deterministic output ordering."""
    if max_views is None or max_views >= len(image_paths):
        return image_paths
    if max_views <= 0:
        raise ValueError("max_views must be positive when set")
    rng = random.Random(seed)
    picked = rng.sample(image_paths, max_views)
    return sorted(picked)


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
            all_feats.append(part_feat[vi])
            all_pts3d.append(pts3d[vi])
        logger.info("  Processed views %d-%d / %d", start, end, len(image_paths))

    return torch.stack(all_feats, dim=0), torch.stack(all_pts3d, dim=0)


# ---------------------------------------------------------------------------
# 3D KNN feature smoothing (PyG + torch_scatter on GPU/CPU; SciPy fallback)
# ---------------------------------------------------------------------------

def _knn_smooth_features_scipy(
    pts3d: torch.Tensor,
    feat: torch.Tensor,
    k: int,
    query_chunk: int = 0,
) -> torch.Tensor:
    from scipy.spatial import cKDTree

    dev, dtype = feat.device, feat.dtype
    V, D, H, W = feat.shape
    N = V * H * W

    points_np = pts3d.detach().cpu().permute(0, 2, 3, 1).reshape(N, 3).numpy()
    feat_np = feat.detach().cpu().permute(0, 2, 3, 1).reshape(N, D).numpy()

    t0 = time.perf_counter()
    logger.info(
        "Building cKDTree for %d points (k=%d), %s ...",
        N,
        k,
        _mem_suffix(),
    )
    tree = cKDTree(points_np)
    logger.info(
        "cKDTree build done in %.1fs, %s",
        time.perf_counter() - t0,
        _mem_suffix(),
    )

    if query_chunk <= 0 or query_chunk >= N:
        logger.info(
            "scipy KNN query single pass (all %d points), %s ...",
            N,
            _mem_suffix(),
        )
        tq = time.perf_counter()
        _, indices = tree.query(points_np, k=k + 1)
        indices = indices[:, 1:]
        smoothed = feat_np[indices].mean(axis=1)
        dt = time.perf_counter() - tq
        logger.info(
            "scipy KNN query done: %d points in %.1fs (~%.0f px/s), %s",
            N,
            dt,
            N / dt if dt > 0 else 0.0,
            _mem_suffix(),
        )
    else:
        smooth_dtype = np.float32 if feat_np.dtype == np.float32 else np.float64
        smoothed = np.empty((N, D), dtype=smooth_dtype)
        n_chunks = (N + query_chunk - 1) // query_chunk
        logger.info(
            "scipy KNN chunked query: %d chunks (chunk_size=%d), ~%d points/chunk, %s",
            n_chunks,
            query_chunk,
            query_chunk,
            _mem_suffix(),
        )
        t_loop = time.perf_counter()
        last_log = t_loop
        for ci, start in enumerate(range(0, N, query_chunk), start=1):
            end = min(start + query_chunk, N)
            _, indices = tree.query(points_np[start:end], k=k + 1)
            indices = indices[:, 1:]
            smoothed[start:end] = feat_np[indices].mean(axis=1).astype(smooth_dtype, copy=False)

            now = time.perf_counter()
            if (
                ci % 10 == 0
                or ci == n_chunks
                or ci == 1
                or (now - last_log) >= 45.0
            ):
                elapsed = now - t_loop
                pct = 100.0 * end / N
                rate = end / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "scipy KNN query progress: chunk %d/%d, points %s/%s (%.2f%%), "
                    "~%.0f px/s, elapsed %.0fs, %s",
                    ci,
                    n_chunks,
                    f"{end:,}",
                    f"{N:,}",
                    pct,
                    rate,
                    elapsed,
                    _mem_suffix(),
                )
                last_log = now

        dt = time.perf_counter() - t_loop
        logger.info(
            "scipy KNN chunked query done: %d chunks in %.1fs (~%.0f px/s avg), %s",
            n_chunks,
            dt,
            N / dt if dt > 0 else 0.0,
            _mem_suffix(),
        )

    out = torch.from_numpy(np.asarray(smoothed)).to(dtype=dtype, device=dev)
    out = out.reshape(V, H, W, D).permute(0, 3, 1, 2)
    return out


def _knn_smooth_features_scipy_per_view(
    pts3d: torch.Tensor,
    feat: torch.Tensor,
    k: int,
) -> torch.Tensor:
    from scipy.spatial import cKDTree

    dev, dtype = feat.device, feat.dtype
    V, D, H, W = feat.shape
    pixels_per_view = H * W
    out = torch.empty((V, D, H, W), dtype=dtype, device=dev)
    logger.info(
        "scipy per-view KNN: %d views × %d px/view (k=%d), %s",
        V,
        pixels_per_view,
        k,
        _mem_suffix(),
    )
    t_all = time.perf_counter()
    last_log = t_all
    for v in range(V):
        pts = pts3d[v].detach().cpu().permute(1, 2, 0).reshape(H * W, 3).numpy()
        f_np = feat[v].detach().cpu().permute(1, 2, 0).reshape(H * W, D).numpy()
        tree = cKDTree(pts)
        _, indices = tree.query(pts, k=k + 1)
        indices = indices[:, 1:]
        smooth_v = f_np[indices].mean(axis=1)
        t = torch.from_numpy(smooth_v).to(dtype=dtype, device=dev).reshape(H, W, D).permute(2, 0, 1)
        out[v] = t
        now = time.perf_counter()
        if v == 0 or v + 1 == V or (v + 1) % max(1, V // 20) == 0 or now - last_log >= 45.0:
            elapsed = now - t_all
            rate = ((v + 1) * pixels_per_view) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "scipy per-view progress: view %d/%d (%.1f%%), ~%.0f px/s, elapsed %.0fs, %s",
                v + 1,
                V,
                100.0 * (v + 1) / V,
                rate,
                elapsed,
                _mem_suffix(),
            )
            last_log = now
    logger.info(
        "scipy per-view KNN done in %.1fs (%d views), %s",
        time.perf_counter() - t_all,
        V,
        _mem_suffix(),
    )
    return out


def _knn_smooth_features_pyg(
    pts3d: torch.Tensor,
    feat: torch.Tensor,
    k: int,
    *,
    per_view: bool = False,
) -> torch.Tensor:
    from torch_geometric.nn import knn_graph
    from torch_scatter import scatter_mean

    V, D, H, W = feat.shape
    device = pts3d.device
    N = V * H * W

    points_flat = pts3d.permute(0, 2, 3, 1).reshape(N, 3).float().contiguous()
    features_flat = feat.permute(0, 2, 3, 1).reshape(N, D).float().contiguous()

    if per_view:
        batch = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(H * W)
        logger.info(
            "PyG knn_graph per-view batched (%d clouds x %d px, k=%d) on %s ...",
            V,
            H * W,
            k,
            device,
        )
        edge_index = knn_graph(points_flat, k=k, batch=batch, loop=False)
    else:
        logger.info("PyG knn_graph for %d points (k=%d) on %s ...", N, k, device)
        edge_index = knn_graph(points_flat, k=k, batch=None, loop=False)

    source, center = edge_index
    neighbor_feat = features_flat[source]
    smoothed_flat = scatter_mean(neighbor_feat, center, dim=0, dim_size=N)

    return smoothed_flat.view(V, H, W, D).permute(0, 3, 1, 2).to(dtype=feat.dtype)


def knn_smooth_features(
    pts3d: torch.Tensor,
    feat: torch.Tensor,
    k: int = 20,
    *,
    scope: str = "global",
    query_chunk: int = 262_144,
    backend: str = "auto",
    pyg_point_threshold: int = PYG_KNN_POINT_THRESHOLD,
) -> torch.Tensor:
    """Smooth instance features by averaging k nearest neighbours in 3D space.

    *global* scope: neighbours may come from any view (single fused point cloud ``V*H*W``).
    *per_view* scope: neighbours are taken within the same image only.

    backends:
      * ``auto``: use PyG for global kNN when point count ``<= pyg_point_threshold`` and PyG loads;
        otherwise SciPy chunked query on CPU. Falls back from PyG to SciPy on errors.
      * ``pyg``: force PyG (can OOM on very large global clouds).
      * ``scipy``: CPU cKDTree; use ``query_chunk`` > 0 to cap peak memory for global kNN.

    Args:
        pts3d: [V, 3, H, W] world coordinates.
        feat:  [V, D, H, W] instance features (should already be L2-normalised).
        k:     Number of 3D nearest neighbours.
        scope: ``"global"`` or ``"per_view"``.
        query_chunk: SciPy global query batch size; ``0`` = one ``tree.query`` for all rows
            (legacy, high RAM for large ``N``).
        backend: ``"auto"``, ``"pyg"``, or ``"scipy"``.
        pyg_point_threshold: In ``auto`` + global, skip PyG when ``N`` exceeds this.

    Returns:
        Smoothed features [V, D, H, W] on the same device as *feat*.
    """
    if scope not in ("global", "per_view"):
        raise ValueError(f"scope must be 'global' or 'per_view', got {scope!r}")
    if backend not in ("auto", "pyg", "scipy"):
        raise ValueError(f"backend must be auto, pyg, or scipy, got {backend!r}")

    V, D, H, W = feat.shape
    N = V * H * W
    per_view = scope == "per_view"

    effective_chunk = query_chunk
    if backend == "auto" and not per_view and N > pyg_point_threshold:
        if effective_chunk <= 0:
            effective_chunk = KNN_QUERY_CHUNK_AUTO_FALLBACK
        logger.info(
            "Point count %d > pyg_threshold %d — using scipy + chunked query (chunk=%d)",
            N,
            pyg_point_threshold,
            effective_chunk,
        )
        return _knn_smooth_features_scipy(pts3d, feat, k, query_chunk=effective_chunk)

    if backend == "scipy":
        return (
            _knn_smooth_features_scipy_per_view(pts3d, feat, k)
            if per_view
            else _knn_smooth_features_scipy(pts3d, feat, k, query_chunk=query_chunk)
        )

    if per_view:
        try:
            return _knn_smooth_features_pyg(pts3d, feat, k, per_view=True)
        except ImportError as err:
            logger.warning("torch_geometric / torch_scatter unavailable (%s); using scipy per-view", err)
            return _knn_smooth_features_scipy_per_view(pts3d, feat, k)
        except Exception as err:
            logger.warning("PyG per-view KNN failed (%s); falling back to scipy per-view", err)
            return _knn_smooth_features_scipy_per_view(pts3d, feat, k)

    if backend == "pyg":
        return _knn_smooth_features_pyg(pts3d, feat, k, per_view=False)

    try:
        return _knn_smooth_features_pyg(pts3d, feat, k, per_view=False)
    except ImportError as err:
        logger.warning("torch_geometric / torch_scatter unavailable (%s); using scipy cKDTree", err)
        ec = query_chunk if query_chunk > 0 else KNN_QUERY_CHUNK_AUTO_FALLBACK
        return _knn_smooth_features_scipy(pts3d, feat, k, query_chunk=ec if N > 1_000_000 else query_chunk)
    except Exception as err:
        logger.warning("PyG KNN failed (%s); falling back to scipy cKDTree", err)
        ec = query_chunk if query_chunk > 0 else KNN_QUERY_CHUNK_AUTO_FALLBACK
        return _knn_smooth_features_scipy(pts3d, feat, k, query_chunk=ec if N > 1_000_000 else query_chunk)


# ---------------------------------------------------------------------------
# Clustering (HDBSCAN on CPU)
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
        id_maps: [V, H, W] int32. Label 0 is reserved for background/ignore;
        clustered pixels use contiguous IDs starting from 1.
    """
    from src.instseg.hdbscan_assign import hdbscan_assign

    V, D, H, W = feat_vdhw.shape
    all_pixels = (
        feat_vdhw.detach().cpu().permute(0, 2, 3, 1).contiguous().reshape(-1, D).float().numpy()
    )

    labels = hdbscan_assign(
        all_pixels,
        cluster_selection_epsilon=eps,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_points=max_cluster_points,
        rng_seed=0,
    )
    # hdbscan_assign follows HDBSCAN/sklearn cluster IDs (0..K-1 after
    # denoising/relabeling). Reserve 0 for background/ignore at this boundary.
    return (labels + 1).reshape(V, H, W)


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

    return (proj.view(V, H, W, 3).detach().cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lut = rng.integers(32, 255, size=(num_colors, 3), dtype=np.uint8)
    if num_colors > 0:
        lut[0] = 0
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
    parser.add_argument(
        "--knn_scope",
        type=str,
        default="global",
        choices=("global", "per_view"),
        help="global: kNN over all views/pixels; per_view: kNN within each view only (faster)",
    )
    parser.add_argument(
        "--knn_backend",
        type=str,
        default="auto",
        choices=("auto", "pyg", "scipy"),
        help="auto: PyG for modest global clouds; scipy for huge N or fallback",
    )
    parser.add_argument(
        "--knn_query_chunk",
        type=int,
        default=262_144,
        help="SciPy global KNN: query this many points at a time (0 = single query, high RAM)",
    )
    parser.add_argument("--max_views", type=int, default=None,
                        help="Randomly use at most this many images (seed=--seed); None = all")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --max_views subsampling")
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
    n_found = len(image_paths)
    image_paths = subsample_image_paths(image_paths, args.max_views, args.seed)
    if len(image_paths) < n_found:
        logger.info(
            "Subsampled images: %d -> %d views (max_views=%s, seed=%d)",
            n_found,
            len(image_paths),
            args.max_views,
            args.seed,
        )
    model = load_model(args.model_path, args.device)

    logger.info("Running inference (image_size=%dx%d) ...", iggt_w, iggt_h)
    feat, pts3d = run_inference(model, image_paths, image_size,
                                device=args.device, batch_size=args.batch_size)
    # feat: [V, D, H, W], pts3d: [V, 3, H, W]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # L2 normalise part features
    V, D, H, W = feat.shape
    feat = F.normalize(feat.float(), dim=1)

    # 3D KNN feature smoothing
    if args.knn_k > 0:
        logger.info(
            "3D KNN smoothing (k=%d, scope=%s, backend=%s) ...",
            args.knn_k,
            args.knn_scope,
            args.knn_backend,
        )
        feat = knn_smooth_features(
            pts3d,
            feat,
            k=args.knn_k,
            scope=args.knn_scope,
            query_chunk=args.knn_query_chunk,
            backend=args.knn_backend,
        )

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

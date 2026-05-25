"""Write instascene-style PNG tree from seg3d split PLYs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.trace_cameras import load_trace_cameras

from .cluster_paths import list_cluster_split_plys
from .job import Seg3dVlmExportJob
from .trace_rasterize import (
    load_gaussians_from_ply,
    raster_gaussian_rgb_u8,
    resolve_trace_backend,
)


@dataclass(slots=True)
class ExportStats:
    wrote: int
    skipped_existing: int


def export_seg3d_vlm_png_tree(job: Seg3dVlmExportJob) -> ExportStats:
    """Render each ``cluster_*.ply`` at each camera → ``out_root/<stem>/cluster_NNN.png``."""
    cameras = load_trace_cameras(
        job.camera_backend,
        source_path=str(job.source_path),
        images_folder=job.images_folder,
        resolution=int(job.render_resolution),
        transforms_json=job.transforms_json,
        transforms_axis=job.transforms_axis,
    )
    if not cameras:
        raise RuntimeError("load_trace_cameras returned no cameras")

    if job.max_views > 0:
        rng = random.Random(int(job.seed))
        n_take = min(job.max_views, len(cameras))
        cameras = rng.sample(cameras, n_take)

    pairs = list_cluster_split_plys(job.splits_dir)
    if not pairs:
        raise FileNotFoundError(f"No cluster_*.ply under {job.splits_dir}")

    if job.seg3d_bg == "white":
        bg = (1.0, 1.0, 1.0)
    else:
        bg = (0.0, 0.0, 0.0)

    job.out_root.mkdir(parents=True, exist_ok=True)

    total = len(cameras) * len(pairs)
    skipped = 0
    wrote = 0
    prog = tqdm(total=total, desc="seg3d_vlm_export", unit="img")
    try:
        for cluster_index, ply_path in pairs:
            means, quats, scales, opacities, colors = load_gaussians_from_ply(ply_path)
            tb = resolve_trace_backend(job.trace_backend, int(scales.shape[1]))

            for cam in cameras:
                stem = Path(cam.image_name).stem
                sub = job.out_root / stem
                sub.mkdir(parents=True, exist_ok=True)
                dst = sub / f"cluster_{cluster_index:03d}.png"
                if job.skip_existing and dst.is_file():
                    skipped += 1
                    prog.update(1)
                    continue

                rgb_np = raster_gaussian_rgb_u8(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                    cam,
                    trace_backend=tb,
                    bg_color_rgb=bg,
                )
                Image.fromarray(rgb_np).save(dst)
                wrote += 1
                prog.update(1)
    finally:
        prog.close()

    print(
        f"[seg3d_vlm_export] wrote={wrote} skipped_existing={skipped} "
        f"cameras={len(cameras)} clusters={len(pairs)} → {job.out_root}"
    )
    return ExportStats(wrote=wrote, skipped_existing=skipped)

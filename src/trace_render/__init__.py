"""Shared Gaussian ``trace()`` rasterization helpers (2DGS / 3DGS).

复用边界（参见 before-implement）：

- ``load_gaussian_ply`` / convention 日志：:mod:`src.coord`
- Surfel / 3DGS ``trace()``：本包 ``trace_rasterize``（与训练侧 ``cuda_splatting`` 前向分离）
"""

from __future__ import annotations

from .cluster_paths import list_cluster_split_plys
from .export_tree import ExportStats, export_seg3d_vlm_png_tree
from .job import Seg3dVlmExportJob, load_job
from .trace_rasterize import (
    TRACE_CHANNELS,
    effective_trace_backend,
    load_gaussians_from_ply,
    raster_gaussian_rgb_u8,
    resolve_trace_backend,
    trace_single_view,
)

__all__ = [
    "TRACE_CHANNELS",
    "ExportStats",
    "Seg3dVlmExportJob",
    "effective_trace_backend",
    "export_seg3d_vlm_png_tree",
    "list_cluster_split_plys",
    "load_gaussians_from_ply",
    "load_job",
    "raster_gaussian_rgb_u8",
    "resolve_trace_backend",
    "trace_single_view",
]

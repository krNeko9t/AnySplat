"""Dispatch loading TraceCamera lists by backend name."""

from __future__ import annotations

from typing import Callable

from .load_colmap import load_colmap_cameras
from .load_transforms_json import (
    load_transforms_json_cameras,
    resolve_transforms_json_path,
)
from .trace_camera import TraceCamera

CameraLoader = Callable[..., list[TraceCamera]]


def _load_colmap_backend(
    *,
    source_path: str,
    images_folder: str,
    resolution: int,
    transforms_json: str | None = None,
    transforms_axis: str = "nerfstudio",
) -> list[TraceCamera]:
    _ = transforms_json, transforms_axis
    return load_colmap_cameras(source_path, images_folder, resolution)


def _load_nerf_transforms_backend(
    *,
    source_path: str,
    images_folder: str,
    resolution: int,
    transforms_json: str | None = None,
    transforms_axis: str = "nerfstudio",
) -> list[TraceCamera]:
    _ = images_folder  # image paths come from JSON ``file_path``
    path = resolve_transforms_json_path(source_path, transforms_json)
    return load_transforms_json_cameras(path, resolution, transforms_axis)


CAMERA_BACKENDS: dict[str, CameraLoader] = {
    "colmap": _load_colmap_backend,
    "nerf_transforms": _load_nerf_transforms_backend,
}


def load_trace_cameras(
    backend: str,
    *,
    source_path: str,
    images_folder: str = "images",
    resolution: int = 1,
    transforms_json: str | None = None,
    transforms_axis: str = "nerfstudio",
) -> list[TraceCamera]:
    """Load cameras for trace / optional render paths.

    Parameters
    ----------
    backend
        ``colmap`` | ``nerf_transforms``.
    source_path
        Scene root (COLMAP sparse parent, or transforms JSON directory).
    images_folder
        COLMAP only: subdirectory under ``source_path`` for RGB images.
    resolution
        Same scaling rule as the legacy COLMAP loader.
    transforms_json
        Optional explicit path when backend is ``nerf_transforms``.
    transforms_axis
        Passed to :func:`load_transforms_json_cameras`.
    """
    key = backend.strip().lower()
    if key not in CAMERA_BACKENDS:
        raise ValueError(
            f"Unknown camera_backend '{backend}', "
            f"expected one of {sorted(CAMERA_BACKENDS)}"
        )
    return CAMERA_BACKENDS[key](
        source_path=source_path,
        images_folder=images_folder,
        resolution=resolution,
        transforms_json=transforms_json,
        transforms_axis=transforms_axis,
    )

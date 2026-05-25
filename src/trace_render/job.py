"""JSON job spec for :func:`~src.trace_render.export_tree.export_seg3d_vlm_png_tree`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Seg3dVlmExportJob:
    source_path: Path
    camera_backend: str
    splits_dir: Path
    out_root: Path
    images_folder: str = "images"
    transforms_json: str | None = None
    transforms_axis: str = "nerfstudio"
    render_resolution: int = 1
    seg3d_bg: str = "black"
    trace_backend: str = "auto"
    seed: int = 0
    max_views: int = 0
    skip_existing: bool = False


def load_job(path: Path | str) -> Seg3dVlmExportJob:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("job file must contain a JSON object")

    def req_str(key: str) -> str:
        v = data.get(key)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"job requires non-empty string field: {key}")
        return v.strip()

    def opt_str(key: str, default: str = "") -> str:
        v = data.get(key, default)
        if v is None:
            return default
        if not isinstance(v, str):
            raise ValueError(f"job.{key} must be a string")
        return v.strip()

    def opt_bool(key: str, default: bool) -> bool:
        v = data.get(key, default)
        if isinstance(v, bool):
            return v
        raise ValueError(f"job.{key} must be boolean")

    def opt_int(key: str, default: int) -> int:
        v = data.get(key, default)
        if isinstance(v, bool):
            raise ValueError(f"job.{key} must be integer")
        if isinstance(v, int):
            return v
        raise ValueError(f"job.{key} must be integer")

    bg = opt_str("seg3d_bg", "black").lower()
    if bg not in ("black", "white"):
        raise ValueError("seg3d_bg must be 'black' or 'white'")

    return Seg3dVlmExportJob(
        source_path=Path(req_str("source_path")).expanduser().resolve(),
        camera_backend=req_str("camera_backend"),
        splits_dir=Path(req_str("splits_dir")).expanduser().resolve(),
        out_root=Path(req_str("out_root")).expanduser().resolve(),
        images_folder=opt_str("images_folder", "images") or "images",
        transforms_json=(
            ts if (ts := opt_str("transforms_json", "")) else None
        ),
        transforms_axis=opt_str("transforms_axis", "nerfstudio") or "nerfstudio",
        render_resolution=opt_int("render_resolution", 1),
        seg3d_bg=bg,
        trace_backend=opt_str("trace_backend", "auto") or "auto",
        seed=opt_int("seed", 0),
        max_views=opt_int("max_views", 0),
        skip_existing=opt_bool("skip_existing", False),
    )

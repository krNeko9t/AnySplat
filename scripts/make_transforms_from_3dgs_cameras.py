"""Convert a 3DGS-style ``cameras.json`` into a NeRF-style ``transforms.json``.

Why this exists
---------------
``src/trace_cameras/registry.py`` only knows two backends: ``colmap`` and
``nerf_transforms``. Some scenes (e.g. ``mipnerf360/garden`` under
``/mnt/storage_pool/3dgs-renderer-benchmark``) ship only the ``cameras.json``
that vanilla 3DGS writes next to a trained ``point_cloud.ply`` — no COLMAP
``sparse/``. This script bridges that gap without touching the trace script.

Convention
----------
3DGS ``camera_to_JSON`` writes ``position`` = camera centre in world and
``rotation`` = the 3x3 block of the **camera-to-world** matrix in OpenCV camera
axes (its local variable is misleadingly named ``W2C``). So the emitted
``transform_matrix`` is c2w/OpenCV and must be read with
``--transforms_axis opencv``.

``cx``/``cy`` are not stored by 3DGS; ``TraceCamera`` is a symmetric frustum and
has no principal-point field anyway, so W/2, H/2 is written and is exactly what
the loader would use.

Usage
-----
    python scripts/make_transforms_from_3dgs_cameras.py \
        --cameras_json .../garden/cameras.json \
        --images_dir   .../garden/images \
        --out_dir      /mnt/storage_pool/.../garden_scene
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="Scene dir to create: transforms.json + images symlink")
    ap.add_argument("--image_ext", default=None,
                    help="Force an extension (default: probe the images dir)")
    args = ap.parse_args()

    cams = json.load(open(args.cameras_json, encoding="utf-8"))
    if not isinstance(cams, list) or not cams:
        raise SystemExit(f"{args.cameras_json} is not a non-empty JSON list")

    images_dir = Path(args.images_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    link = out_dir / "images"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            raise SystemExit(f"{link} exists and is not a symlink")
    link.symlink_to(images_dir)

    on_disk = {p.stem: p.name for p in sorted(images_dir.iterdir()) if p.is_file()}

    widths = {int(c["width"]) for c in cams}
    heights = {int(c["height"]) for c in cams}
    fxs = {round(float(c["fx"]), 6) for c in cams}
    fys = {round(float(c["fy"]), 6) for c in cams}
    if len(widths) != 1 or len(heights) != 1 or len(fxs) != 1 or len(fys) != 1:
        raise SystemExit(
            "transforms.json needs one shared intrinsic; this cameras.json has "
            f"w={sorted(widths)} h={sorted(heights)} fx={sorted(fxs)} fy={sorted(fys)}"
        )

    w, h = widths.pop(), heights.pop()
    fx, fy = float(cams[0]["fx"]), float(cams[0]["fy"])

    frames = []
    missing = []
    for c in cams:
        name = c["img_name"]
        fname = (name + args.image_ext) if args.image_ext else on_disk.get(name)
        if fname is None:
            missing.append(name)
            continue
        r = c["rotation"]
        p = c["position"]
        mat = [
            [r[0][0], r[0][1], r[0][2], p[0]],
            [r[1][0], r[1][1], r[1][2], p[1]],
            [r[2][0], r[2][1], r[2][2], p[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append({"file_path": f"images/{fname}", "transform_matrix": mat})

    if missing:
        raise SystemExit(f"{len(missing)} cameras have no image on disk, e.g. {missing[:5]}")

    data = {
        "camera_model": "PINHOLE",
        "fl_x": fx, "fl_y": fy,
        "cx": w / 2.0, "cy": h / 2.0,
        "w": w, "h": h,
        "frames": frames,
    }
    out_json = out_dir / "transforms.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

    print(f"Wrote {out_json}")
    print(f"  {len(frames)} frames, {w}x{h}, fl_x={fx:.3f} fl_y={fy:.3f}")
    print(f"  images -> {images_dir}")
    print("  Load with: --camera_backend nerf_transforms --transforms_axis opencv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_index(name: str) -> int | None:
    # Matches e.g. Image_0_0_0001_0.png, Depth_12_0_0001_0.npy, camview_99_0_0001_0.npz
    m = re.search(r"_(\d+)_0_0001_0\.", name)
    if not m:
        # camview has slightly different prefix but same pattern
        m = re.search(r"camview_(\d+)_0_0001_0\.", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True, help="Path to infinigen scene (contains frames/)")
    ap.add_argument("--camera", type=str, default="camera_0")
    ap.add_argument("--out", type=str, default=None, help="Output manifest path (.jsonl). Defaults to <scene_dir>/manifest.jsonl")
    ap.add_argument("--use_depth", choices=["npy", "png"], default="npy")
    ap.add_argument("--use_seg", choices=["npy", "png"], default="npy")
    ap.add_argument("--scene_id", type=str, default=None)
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    frames = scene_dir / "frames"
    cam = args.camera

    img_dir = frames / "Image" / cam
    depth_dir = frames / "Depth" / cam
    seg_dir = frames / "ObjectSegmentation" / cam
    camview_dir = frames / "camview" / cam

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(depth_dir)
    if not seg_dir.exists():
        raise FileNotFoundError(seg_dir)
    if not camview_dir.exists():
        raise FileNotFoundError(camview_dir)

    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        raise ValueError(f"No images found in {img_dir}")

    depth_files = sorted(depth_dir.glob(f"*.{args.use_depth}"))
    seg_files = sorted(seg_dir.glob(f"*.{args.use_seg}"))
    camviews = sorted(camview_dir.glob("*.npz"))
    if not depth_files:
        raise ValueError(f"No depth {args.use_depth} found in {depth_dir}")
    if not seg_files:
        raise ValueError(f"No seg {args.use_seg} found in {seg_dir}")
    if not camviews:
        raise ValueError(f"No camview npz found in {camview_dir}")

    def build_map(paths):
        m = {}
        for p in paths:
            idx = parse_index(p.name)
            if idx is None:
                continue
            m[idx] = p
        return m

    img_map = build_map(imgs)
    depth_map = build_map(depth_files)
    seg_map = build_map(seg_files)
    cam_map = {}
    for p in camviews:
        m = re.search(r"camview_(\d+)_0_0001_0\.npz", p.name)
        if m:
            cam_map[int(m.group(1))] = p

    common = sorted(set(img_map) & set(depth_map) & set(seg_map) & set(cam_map))
    if not common:
        raise ValueError("No common indices across Image/Depth/Seg/camview.")

    scene_id = args.scene_id or scene_dir.name
    out_path = Path(args.out) if args.out is not None else (scene_dir / "manifest.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_out = []
    for idx in common:
        cam_npz = np.load(cam_map[idx])
        K = cam_npz["K"].astype(np.float32).tolist()
        T = cam_npz["T"].astype(np.float32).tolist()  # treated as c2w
        HW = cam_npz["HW"].astype(int).tolist()

        frames_out.append(
            {
                "rgb_path": str(img_map[idx].relative_to(scene_dir)),
                "depth_path": str(depth_map[idx].relative_to(scene_dir)),
                "instance_mask_path": str(seg_map[idx].relative_to(scene_dir)),
                "K_px": K,
                "c2w": T,
                "HW": HW,
                "near": args.near,
                "far": args.far,
            }
        )

    scene_obj = {"scene_id": scene_id, "frames": frames_out}
    with out_path.open("w") as f:
        f.write(json.dumps(scene_obj) + "\n")

    print(f"Wrote {len(frames_out)} frames to {out_path}")


if __name__ == "__main__":
    main()


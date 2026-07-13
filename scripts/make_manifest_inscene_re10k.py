#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_scenes(root: Path) -> Iterable[Path]:
    """Yield per-scene directories under processed_re10k root.

    这里假设目录结构类似：
      processed_re10k/<scene_id>/
        rgb/              # RGB 图像
        refined_ins_ids/  # 实例 id map
        cam/              # 多个 npz，每个包含单个视角的 intrinsics / pose
    请根据你实际的 RE10K 预处理结果调整。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def build_scene_manifest(scene_dir: Path, near: float, far: float) -> dict:
    """Build one scene manifest dict for RE10K-like data (no depth)."""
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name

    rgb_dir = scene_dir / "rgb"
    inst_dir = scene_dir / "refined_ins_ids"
    cam_dir = scene_dir / "cam"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not inst_dir.is_dir():
        raise FileNotFoundError(inst_dir)
    if not cam_dir.is_dir():
        raise FileNotFoundError(cam_dir)

    rgbs = sorted(rgb_dir.iterdir())
    insts = sorted(inst_dir.iterdir())
    cams = sorted(cam_dir.glob("*.npz"))

    if not rgbs:
        raise ValueError(f"No rgb frames found in {rgb_dir}")
    if len(rgbs) != len(insts) or len(rgbs) != len(cams):
        raise ValueError(
            f"rgb/inst/cam count mismatch in {scene_dir}: "
            f"{len(rgbs)} rgb, {len(insts)} masks, {len(cams)} cams"
        )

    frames_out = []
    for rgb_path, inst_path, cam_path in zip(rgbs, insts, cams):
        cam = np.load(cam_path)
        keys = set(cam.keys())
        if {"intrinsic", "pose"} <= keys:
            K = cam["intrinsic"]
            T = cam["pose"]
        else:
            raise ValueError(
                f"Unsupported cam npz keys in {cam_path}: expected at least 'intrinsic' and 'pose', got {list(keys)}"
            )

        # Upgrade pose to 4x4 if needed.
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float32)
            T4[:3, :] = T.astype(np.float32)
            T = T4
        else:
            T = T.astype(np.float32)

        import PIL.Image as Image

        with Image.open(rgb_path) as im:
            w, h = im.size
        HW = [int(h), int(w)]

        frame = {
            "rgb_path": str(rgb_path.relative_to(scene_dir)),
            "instance_mask_path": str(inst_path.relative_to(scene_dir)),
            # RE10K 无 depth：不写 depth_path，让 DatasetManifest 走“无深度”分支。
            "K_px": K.astype(np.float32).tolist(),
            "c2w": T.tolist(),
            "HW": HW,
            "near": float(near),
            "far": float(far),
        }
        frames_out.append(frame)

    return {"scene_id": scene_id, "frames": frames_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root dir of processed_re10k (per-scene subdirs with rgb/refined_ins_ids/cam).",
    )
    ap.add_argument(
        "--manifest_out",
        type=Path,
        default=None,
        help="Output manifest .jsonl path; defaults to <data_root>/../manifest_re10k.jsonl",
    )
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_path: Path = args.manifest_out or (data_root.parent / "manifest_re10k.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = list(iter_scenes(data_root))
    if not scenes:
        raise ValueError(f"No scene directories found under {data_root}")

    num_frames_total = 0
    with out_path.open("w") as f:
        for scene_dir in scenes:
            scene_obj = build_scene_manifest(scene_dir, near=args.near, far=args.far)
            num_frames_total += len(scene_obj.get("frames", []))
            f.write(json.dumps(scene_obj) + "\n")

    print(f"Wrote {len(scenes)} scenes and {num_frames_total} frames to {out_path}")


if __name__ == "__main__":
    main()


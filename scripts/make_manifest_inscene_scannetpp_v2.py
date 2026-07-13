#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_scenes(root: Path) -> Iterable[Path]:
    """Yield per-scene directories under processed_scannetpp_v2 root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def load_metadata(scene_dir: Path) -> dict:
    """Load camera intrinsics / trajectories from scene-level metadata.

    Prefer iPhone metadata (matched to images/ resolution). Fall back to DSLR if needed.
    """
    candidates = [
        scene_dir / "scene_iphone_metadata.npz",
        scene_dir / "scene_dslr_metadata.npz",
        scene_dir / "scene_metadata.npz",
    ]
    for c in candidates:
        if c.exists():
            data = np.load(c)
            return {"path": c, "data": data}
    raise FileNotFoundError(f"No scene_*_metadata.npz found in {scene_dir}")


def build_scene_manifest(scene_dir: Path, data_root: Path, near: float, far: float) -> dict:
    """Build one scene manifest dict compatible with DatasetManifest.

    Expected directory layout per scene (under processed_scannetpp_v2/<scene_id>/):
      - images/             # RGB frames
      - depth/              # depth maps (optional but expected here)
      - refined_ins_ids/    # instance id maps
      - scene_*_metadata.npz  # contains intrinsics / trajectories arrays
    """
    scene_dir = Path(scene_dir)
    data_root = Path(data_root)
    scene_id = scene_dir.name

    img_dir = scene_dir / "images"
    depth_dir = scene_dir / "depth"
    inst_dir = scene_dir / "refined_ins_ids"

    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)
    if not inst_dir.is_dir():
        raise FileNotFoundError(inst_dir)

    meta = load_metadata(scene_dir)
    data = meta["data"]
    keys = set(data.keys())

    # Try common key names; adjust here if your metadata uses different ones.
    if {"intrinsics", "trajectories", "images"} <= keys:
        intrinsics = data["intrinsics"]  # [N,3,3]
        trajectories = data["trajectories"]  # [N,4,4] or [N,3,4]
        image_names = data["images"]  # [N] array of filenames (e.g. "frame_000000.jpg" or "DSC07231.jpg")
    else:
        raise ValueError(
            f"Unsupported metadata keys in {meta['path']}: "
            f"expected at least 'intrinsics', 'trajectories', 'images', got {list(keys)}"
        )

    num_views_meta = intrinsics.shape[0]
    if len(image_names) != num_views_meta:
        raise ValueError(
            f"metadata images ({len(image_names)}) and intrinsics ({num_views_meta}) length mismatch in {scene_dir}"
        )

    # Pre-index instance masks by stem, but只保留与 RGB 对齐的 frame_*.jpg.npy。
    mask_map: dict[str, Path] = {}
    for p in inst_dir.iterdir():
        if not p.name.endswith(".jpg.npy"):
            continue
        stem = p.name.replace(".jpg.npy", "")  # e.g. "frame_000000"
        mask_map[stem] = p

    frames_out = []
    for idx in range(num_views_meta):
        # Use metadata ordering; skip views without both RGB and mask.
        name = image_names[idx]
        # Expect filenames like "frame_000000.jpg". 只保留有实例掩码的 frame_* 视角。
        stem = Path(name).stem  # "frame_000000" or "DSC07231"
        if not stem.startswith("frame_"):
            continue

        img_path = img_dir / name
        if not img_path.exists():
            continue

        inst_path = mask_map.get(stem)
        if inst_path is None or not inst_path.exists():
            continue

        depth_path = None
        if depth_dir.exists():
            cand = depth_dir / f"{stem}.png"
            if cand.exists():
                depth_path = cand

        K = intrinsics[idx].astype(np.float32).tolist()
        T = trajectories[idx]
        # If trajectories is [N,3,4], upgrade to 4x4.
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float32)
            T4[:3, :] = T.astype(np.float32)
            T = T4
        else:
            T = T.astype(np.float32)

        # Load image once to get HW; if that is too slow, you can stash HW elsewhere.
        import PIL.Image as Image

        with Image.open(img_path) as im:
            w, h = im.size
        HW = [int(h), int(w)]

        # Paths should be relative to data_root (processed_scannetpp_v2),
        # so that DatasetManifest.root can be set to data_root.
        scene_rel = scene_dir.relative_to(data_root)
        frame = {
            "rgb_path": str((scene_rel / img_path.relative_to(scene_dir))),
            "instance_mask_path": str((scene_rel / inst_path.relative_to(scene_dir))),
            "K_px": K,
            "c2w": T.tolist(),
            "HW": HW,
            "near": float(near),
            "far": float(far),
        }
        if depth_path is not None:
            frame["depth_path"] = str((scene_rel / depth_path.relative_to(scene_dir)))

        frames_out.append(frame)

    return {"scene_id": scene_id, "frames": frames_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root dir of processed_scannetpp_v2 (e.g. InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2)",
    )
    ap.add_argument(
        "--manifest_out",
        type=Path,
        default=None,
        help="Output manifest .jsonl path; defaults to <data_root>/manifest_scannetpp_v2.jsonl",
    )
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_path: Path = args.manifest_out or (data_root / "manifest_scannetpp_v2.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = list(iter_scenes(data_root))
    if not scenes:
        raise ValueError(f"No scene directories found under {data_root}")

    num_scenes_written = 0
    num_frames_total = 0
    num_scenes_skipped_no_meta = 0
    with out_path.open("w") as f:
        for scene_dir in scenes:
            try:
                scene_obj = build_scene_manifest(scene_dir, data_root=data_root, near=args.near, far=args.far)
            except FileNotFoundError as e:
                # e.g. missing scene_*_metadata.npz; skip this scene.
                print(f"Skipping scene {scene_dir} due to missing metadata: {e}")
                num_scenes_skipped_no_meta += 1
                continue

            frames = scene_obj.get("frames", [])
            if not frames:
                # Nothing usable for this scene (e.g. no matching frame_* views with masks).
                continue

            f.write(json.dumps(scene_obj) + "\n")
            num_scenes_written += 1
            num_frames_total += len(frames)

    print(
        f"Wrote {num_scenes_written} scenes and {num_frames_total} frames to {out_path} "
        f"(skipped {num_scenes_skipped_no_meta} scenes without metadata)."
    )


if __name__ == "__main__":
    main()


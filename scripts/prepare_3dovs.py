#!/usr/bin/env python3
"""Convert a 3dovs scene into the manifest format expected by DatasetCustom.

Usage:
    python scripts/prepare_3dovs.py --scene_dir 3dovs/bench

Outputs (written into scene_dir):
    - manifest.jsonl   (one line, one scene)
    - physics_labels.json  (instance_id -> label string)
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP binary parsers
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> dict:
    """Parse cameras.bin → {camera_id: {model, width, height, params}}."""
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(n_params * 8))
            cameras[cam_id] = dict(model_id=model_id, width=width, height=height, params=params)
    return cameras


def _read_images_bin(path: Path) -> list[dict]:
    """Parse images.bin → list of {image_id, camera_id, name, qvec, tvec}."""
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)
            images.append(dict(
                image_id=image_id, camera_id=camera_id,
                name=name.decode(), qvec=qvec, tvec=tvec,
            ))
    images.sort(key=lambda x: x["name"])
    return images


def _qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def _colmap_to_K(cam: dict) -> list[list[float]]:
    """Convert COLMAP camera params to 3x3 intrinsic matrix (ignoring distortion)."""
    model_id = cam["model_id"]
    params = cam["params"]
    w, h = cam["width"], cam["height"]

    if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params
        return [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params
        return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    elif model_id == 2:  # SIMPLE_RADIAL: f, cx, cy, k
        f, cx, cy, _k = params
        return [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    elif model_id == 3:  # RADIAL: f, cx, cy, k1, k2
        f, cx, cy, _k1, _k2 = params
        return [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    elif model_id == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[:4]
        return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model_id}")


def _colmap_to_c2w(img: dict) -> list[list[float]]:
    """Convert COLMAP qvec/tvec (w2c) to 4x4 c2w matrix."""
    R = _qvec_to_rotmat(img["qvec"])
    t = np.array(img["tvec"])
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w.tolist()


# ---------------------------------------------------------------------------
# Physics labels conversion
# ---------------------------------------------------------------------------

BEHAVIOR_MAP = {
    "static": "static",
    "dynamic_rigid": "rigid",
    "dynamic_soft": "soft",
    "out_of_range": "unknown",
}


def _convert_phys_params(phys_path: Path) -> dict[str, str]:
    """Convert phys_params.json → {id_str: label_str}."""
    with open(phys_path) as f:
        items = json.load(f)
    labels = {}
    for item in items:
        inst_id = str(item["id"])
        behavior = item["response"]["behavior_template"]
        labels[inst_id] = BEHAVIOR_MAP.get(behavior, "unknown")
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare 3dovs scene for training")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory (e.g. 3dovs/bench)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_id = scene_dir.name

    # --- Parse COLMAP ---
    sparse_dir = scene_dir / "sparse" / "0"
    cameras = _read_cameras_bin(sparse_dir / "cameras.bin")
    images = _read_images_bin(sparse_dir / "images.bin")

    # --- Build frames ---
    frames = []
    for img in images:
        cam = cameras[img["camera_id"]]
        stem = Path(img["name"]).stem

        id_map_path = f"id_maps/{stem}.npy"
        id_map_full = scene_dir / id_map_path
        if not id_map_full.exists():
            print(f"  WARN: skipping {img['name']}, no id_map at {id_map_path}")
            continue

        frames.append({
            "rgb_path": f"images/{img['name']}",
            "instance_mask_path": id_map_path,
            "K_px": _colmap_to_K(cam),
            "c2w": _colmap_to_c2w(img),
            "HW": [int(cam["height"]), int(cam["width"])],
        })

    print(f"Scene '{scene_id}': {len(frames)} frames with id_maps")

    # --- Physics labels ---
    phys_path = scene_dir / "phys_params.json"
    if phys_path.exists():
        labels = _convert_phys_params(phys_path)
        out_labels = scene_dir / "physics_labels.json"
        with open(out_labels, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"  Wrote {out_labels} ({len(labels)} instances)")
    else:
        labels = None
        print("  WARN: no phys_params.json found")

    # --- Manifest ---
    scene_entry = {"scene_id": scene_id, "frames": frames}
    if labels is not None:
        scene_entry["physics_labels_path"] = "physics_labels.json"

    manifest_path = scene_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        f.write(json.dumps(scene_entry) + "\n")
    print(f"  Wrote {manifest_path}")


if __name__ == "__main__":
    main()

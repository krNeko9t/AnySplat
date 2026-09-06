#!/usr/bin/env python3
"""Build the InsScene-15K / Infinigen physics training manifest.

Walks ``<root>/processed_infinigen/scene_XXX/<hash>`` for frames and pairs each
scene with its VLM pseudo-label JSON under
``<root>/preprocessed/annotations/infinigen/infinigen/scene_XXX/<hash>/<name>``,
attaching it as ``physics_labels_path`` (see
``src/dataset/physics/parsers.py``, key ``instascene_vlm_physgm``).

Scene ids are ``infinigen_<scene_XXX>_<hash>``; frame paths are relative to
``--root`` so the manifest travels with the dataset.

Example:
  python scripts/make_manifest_infinigen_phys.py \
    --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K \
    --out  /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/manifest_infinigen_phys.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from make_manifest_infinigen import make_scene_manifest

ANN_SUBDIR = Path("preprocessed/annotations/infinigen/infinigen")
PROC_SUBDIR = Path("processed_infinigen")


def iter_scene_dirs(proc_root: Path):
    for group in sorted(proc_root.glob("scene_*")):
        if not group.is_dir():
            continue
        for scene in sorted(group.iterdir()):
            if scene.is_dir():
                yield scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="InsScene-15K dataset root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotation_name", type=str, default="Qwen3.6-27B.json")
    ap.add_argument("--camera", type=str, default="camera_0")
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=None, help="Debug: only first N scenes")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    proc_root = root / PROC_SUBDIR
    ann_root = root / ANN_SUBDIR
    if not proc_root.is_dir():
        raise FileNotFoundError(proc_root)
    if not ann_root.is_dir():
        raise FileNotFoundError(ann_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped: dict[str, int] = {}
    total_frames = 0

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with args.out.open("w") as fout:
        for i, scene_dir in enumerate(iter_scene_dirs(proc_root)):
            if args.limit is not None and i >= args.limit:
                break
            rel = scene_dir.relative_to(proc_root)  # scene_XXX/<hash>
            ann_path = ann_root / rel / args.annotation_name
            if not ann_path.is_file():
                skip("missing annotation")
                continue
            try:
                scene_obj = make_scene_manifest(
                    scene_dir,
                    camera=args.camera,
                    near=args.near,
                    far=args.far,
                )
            except (FileNotFoundError, ValueError) as e:
                skip(f"frames: {type(e).__name__}")
                continue
            frames = scene_obj.get("frames") or []
            if not frames:
                skip("no frames")
                continue
            # scene-relative → root-relative
            scene_rel = scene_dir.relative_to(root)
            for fr in frames:
                for key in ("rgb_path", "depth_path", "instance_mask_path"):
                    if key in fr:
                        fr[key] = str(scene_rel / fr[key])
            scene_obj["scene_id"] = "infinigen_" + str(rel).replace("/", "_")
            scene_obj["physics_labels_path"] = str(ann_path.relative_to(root))
            fout.write(json.dumps(scene_obj) + "\n")
            kept += 1
            total_frames += len(frames)
            if kept % 100 == 0:
                print(f"  {kept} scenes...", flush=True)

    print(f"Wrote {kept} scenes / {total_frames} frames → {args.out}")
    if skipped:
        print("Skipped:", skipped)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bridge instascene VLM annotation runs → Anysplat training manifest.jsonl.

Reads an instascene scene-selection lockfile + a VLM output directory, builds
per-scene frames via dataset-specific adapters, and attaches
``physics_labels_path`` pointing at the raw VLM JSON (e.g. Qwen3.6-27B.json).

Does not parse physical_property; that belongs to a PhysicsParser downstream.

Example:
  python scripts/make_manifest_from_instascene_vlm.py \\
    --scene_manifest .../manifests/scannetpp_v2_100.json \\
    --vlm_output_dir .../outputs/run_scannet_100 \\
    --annotation_name Qwen3.6-27B.json \\
    --manifest_out .../manifest_phys_scannet100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from make_manifest_inscene_scannetpp_v2 import build_scene_manifest as build_scannetpp_v2_scene

FrameBuilder = Callable[[Path, Path, float, float], dict]

FRAME_BUILDERS: dict[str, FrameBuilder] = {
    "scannetpp_v2": build_scannetpp_v2_scene,
}


def load_scene_selection(path: Path) -> dict:
    with path.open("r") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    kind = doc.get("kind")
    if kind != "scene_selection":
        raise ValueError(f"Expected kind='scene_selection', got {kind!r} in {path}")
    if "dataset_roots" not in doc or not isinstance(doc["dataset_roots"], dict):
        raise ValueError(f"Missing or invalid dataset_roots in {path}")
    if "scenes" not in doc or not isinstance(doc["scenes"], list):
        raise ValueError(f"Missing or invalid scenes list in {path}")
    return doc


def make_unique_scene_id(dataset_id: str, scene_id: str) -> str:
    return f"{dataset_id}_{scene_id}"


def build_one_scene(
    *,
    dataset_id: str,
    scene_id: str,
    dataset_root: Path,
    vlm_output_dir: Path,
    annotation_name: str,
    near: float,
    far: float,
) -> tuple[dict | None, str | None]:
    builder = FRAME_BUILDERS.get(dataset_id)
    if builder is None:
        return None, f"no FrameBuilder for dataset_id={dataset_id!r}"

    ann_path = (vlm_output_dir / dataset_id / scene_id / annotation_name).resolve()
    if not ann_path.is_file():
        return None, f"missing annotation {ann_path}"

    scene_dir = dataset_root / scene_id
    if not scene_dir.is_dir():
        return None, f"missing scene dir {scene_dir}"

    try:
        scene_obj = builder(scene_dir, dataset_root, near, far)
    except FileNotFoundError as e:
        return None, f"missing metadata/files: {e}"
    except ValueError as e:
        return None, f"bad metadata: {e}"

    frames = scene_obj.get("frames") or []
    if not frames:
        return None, "no usable frames"

    scene_obj["scene_id"] = make_unique_scene_id(dataset_id, scene_id)
    scene_obj["physics_labels_path"] = str(ann_path)
    return scene_obj, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bridge instascene VLM outputs to an Anysplat manifest.jsonl."
    )
    ap.add_argument("--scene_manifest", type=Path, required=True)
    ap.add_argument("--vlm_output_dir", type=Path, required=True)
    ap.add_argument("--annotation_name", type=str, default="Qwen3.6-27B.json")
    ap.add_argument("--manifest_out", type=Path, required=True)
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    scene_manifest = Path(args.scene_manifest)
    vlm_output_dir = Path(args.vlm_output_dir)
    out_path = Path(args.manifest_out)

    if not scene_manifest.is_file():
        raise FileNotFoundError(scene_manifest)
    if not vlm_output_dir.is_dir():
        raise FileNotFoundError(vlm_output_dir)

    doc = load_scene_selection(scene_manifest)
    dataset_roots = {k: Path(v) for k, v in doc["dataset_roots"].items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_frames = 0
    skip_counts: dict[str, int] = {}

    with out_path.open("w") as f:
        for entry in doc["scenes"]:
            if not isinstance(entry, dict):
                skip_counts["bad_entry"] = skip_counts.get("bad_entry", 0) + 1
                continue
            dataset_id = entry.get("dataset_id")
            scene_id = entry.get("scene_id")
            if not dataset_id or not scene_id:
                skip_counts["bad_entry"] = skip_counts.get("bad_entry", 0) + 1
                continue

            root = dataset_roots.get(dataset_id)
            if root is None:
                print(f"Skip {dataset_id}/{scene_id}: not in dataset_roots")
                skip_counts["missing_root"] = skip_counts.get("missing_root", 0) + 1
                continue

            scene_obj, reason = build_one_scene(
                dataset_id=dataset_id,
                scene_id=scene_id,
                dataset_root=root,
                vlm_output_dir=vlm_output_dir,
                annotation_name=args.annotation_name,
                near=args.near,
                far=args.far,
            )
            if scene_obj is None:
                print(f"Skip {dataset_id}/{scene_id}: {reason}")
                key = "other"
                assert reason is not None
                if reason.startswith("no FrameBuilder"):
                    key = "no_builder"
                elif reason.startswith("missing annotation"):
                    key = "missing_annotation"
                elif reason.startswith("missing scene dir"):
                    key = "missing_scene_dir"
                elif reason.startswith("missing metadata"):
                    key = "missing_metadata"
                elif reason.startswith("bad metadata"):
                    key = "bad_metadata"
                elif reason.startswith("no usable frames"):
                    key = "no_frames"
                skip_counts[key] = skip_counts.get(key, 0) + 1
                continue

            f.write(json.dumps(scene_obj) + "\n")
            n_written += 1
            n_frames += len(scene_obj["frames"])

    print(f"Wrote {n_written} scenes / {n_frames} frames → {out_path.resolve()}")
    if skip_counts:
        print(f"Skipped: {skip_counts}")
    print(f"Registered FrameBuilders: [{', '.join(sorted(FRAME_BUILDERS))}]")
    used_ids = {e.get("dataset_id") for e in doc["scenes"] if isinstance(e, dict)}
    used_ids.discard(None)
    if len(used_ids) == 1:
        only = next(iter(used_ids))
        print(f"Hint: dataset.manifest.root={dataset_roots[only].resolve()}")
    print(f"Hint: dataset.manifest.manifest_path={out_path.resolve()}")


if __name__ == "__main__":
    main()

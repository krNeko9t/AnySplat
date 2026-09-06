#!/usr/bin/env python3
"""Emit the Infinigen instance → object-class table for the physics labels.

The VLM pseudo-labels carry no class field. Infinigen ships one: each frame's
``frames/Objects/<camera>/Objects_*.json`` maps ``object_index`` → blender object
name, and ``object_index`` is exactly the value stored in the
``ObjectSegmentation`` masks and in the VLM record's ``id``.

Class is derived from the name:
  ``BedFactory(1961782).spawn_asset(...)`` → ``Bed``
  ``bedroom_0/0.wall``                    → ``room:wall``
  anything else                           → ``other:<name>``

Output JSON: ``{scene_id: {inst_id_str: {"name": ..., "class": ...}}}``, keyed by
the same ``scene_id`` as ``manifest_infinigen_phys.jsonl``.

Example:
  python scripts/build_infinigen_class_table.py \
    --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K \
    --out  /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/infinigen_class_table.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ANN_SUBDIR = Path("preprocessed/annotations/infinigen/infinigen")
PROC_SUBDIR = Path("processed_infinigen")

_FACTORY = re.compile(r"^([A-Za-z0-9_]+)Factory\(")


def derive_class(name: str) -> str:
    m = _FACTORY.match(name)
    if m:
        return m.group(1)
    if "." in name:
        return "room:" + name.rsplit(".", 1)[1]
    return "other:" + name


def scene_index_to_name(objects_dir: Path, max_frames: int) -> dict[int, str]:
    """Union object_index → name over the first ``max_frames`` frames."""
    idx_to_name: dict[int, str] = {}
    for j in sorted(objects_dir.glob("*.json"))[:max_frames]:
        with j.open("r") as f:
            doc = json.load(f)
        for name, meta in doc.items():
            idx = meta.get("object_index")
            if isinstance(idx, int):
                idx_to_name.setdefault(idx, name)
    return idx_to_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotation_name", type=str, default="Qwen3.6-27B.json")
    ap.add_argument("--camera", type=str, default="camera_0")
    ap.add_argument("--max_frames", type=int, default=8,
                    help="Frames per scene unioned for the index→name map")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    proc_root, ann_root = root / PROC_SUBDIR, root / ANN_SUBDIR

    table: dict[str, dict[str, dict[str, str]]] = {}
    cls_counts: collections.Counter[str] = collections.Counter()
    n_records = n_unnamed = 0
    unnamed_ids: collections.Counter[int] = collections.Counter()

    for group in sorted(proc_root.glob("scene_*")):
        for scene_dir in sorted(p for p in group.iterdir() if p.is_dir()):
            rel = scene_dir.relative_to(proc_root)
            ann_path = ann_root / rel / args.annotation_name
            objects_dir = scene_dir / "frames" / "Objects" / args.camera
            if not ann_path.is_file() or not objects_dir.is_dir():
                continue
            idx_to_name = scene_index_to_name(objects_dir, args.max_frames)
            with ann_path.open("r") as f:
                recs = json.load(f)
            entry: dict[str, dict[str, str]] = {}
            for r in recs:
                if not isinstance(r, dict) or "id" not in r:
                    continue
                n_records += 1
                name = idx_to_name.get(int(r["id"]))
                if name is None:
                    n_unnamed += 1
                    unnamed_ids[int(r["id"])] += 1
                    continue
                cls = derive_class(name)
                cls_counts[cls] += 1
                entry[str(r["id"])] = {"name": name, "class": cls}
            table["infinigen_" + str(rel).replace("/", "_")] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(table, f)

    print(f"scenes={len(table)} records={n_records} named={n_records - n_unnamed} "
          f"unnamed={n_unnamed} ({n_unnamed / max(1, n_records):.2%})")
    print("unnamed ids (top 10):", unnamed_ids.most_common(10))
    print(f"distinct classes: {len(cls_counts)}")
    for name, n in cls_counts.most_common(25):
        print(f"  {name:<28} {n}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

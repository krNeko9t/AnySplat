#!/usr/bin/env python3
"""Precompute the **class-lookup baseline** the student has to beat.

Ticket 03 fixes the reference frame: the teacher is not a row (the student's
labels *are* the teacher, so its error is 0 by construction).  What the main
table actually reports is the student's position relative to a class lookup
table -- predict each instance's property as the mean over its class in the
training split.  Ticket 04 measured that this baseline is weak (log10 E MAE
0.890 vs 0.751 for a constant), which is the point: it is the trivial baseline
nobody publishes.

Output, joined ahead of time so the dataset parser does no joins:

    {"by_scene": {scene_id: {inst_id: [z_density, z_youngs, z_poisson]}},
     "meta": {...}}

Values are in the same z-space as ``PhysGMTarget.value_lut`` (see
PHYSGM_NORMALIZATION), so a metric can subtract them directly.  Classes unseen
in the training split fall back to the training mean, i.e. z = 0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dataset.physics.parsers import (  # noqa: E402
    PHYSGM_NORMALIZATION,
    _VLM_PROP_KEYS,
    _iter_vlm_instances,
)
from src.dataset.physics.types import PROPERTY_NAMES  # noqa: E402


def z_values(props: dict) -> list[float] | None:
    out: list[float] = []
    try:
        for key, spec in zip(_VLM_PROP_KEYS, PHYSGM_NORMALIZATION, strict=True):
            v = float(props[key]["mean"]) * spec.si_scale
            if spec.log10:
                v = math.log10(max(v, 1.0))
            out.append((v - spec.mean) / spec.std)
    except (KeyError, TypeError, ValueError):
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--class_table", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="scripts/split_scenes.py output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    class_table = json.loads(args.class_table.read_text())
    split = json.loads(args.split.read_text())
    train_ids = set(split["train_scene_ids"])

    scenes: dict[str, dict] = {}
    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                scenes[str(d["scene_id"])] = d

    # ---- per-instance z, and the training-split class means -----------------
    per_scene: dict[str, dict[str, list[float]]] = {}
    by_class: dict[str, list[list[float]]] = defaultdict(list)
    n_no_class = 0
    for sid, scene in scenes.items():
        labels_path = Path(scene["physics_labels_path"])
        if not labels_path.is_absolute():
            labels_path = args.root / labels_path
        if not labels_path.exists():
            continue
        raw = json.loads(labels_path.read_text())
        table = class_table.get(sid, {})
        vals: dict[str, list[float]] = {}
        for inst_id, props in _iter_vlm_instances(raw):
            z = z_values(props)
            if z is None:
                continue
            vals[str(inst_id)] = z
            entry = table.get(str(inst_id))
            if entry is None:
                n_no_class += 1
                continue
            if sid in train_ids:
                by_class[entry["class"]].append(z)
        per_scene[sid] = vals

    class_mean = {
        cls: [statistics.fmean(z[i] for z in rows) for i in range(len(PROPERTY_NAMES))]
        for cls, rows in by_class.items()
    }

    # ---- join: scene -> inst -> class mean ---------------------------------
    zero = [0.0] * len(PROPERTY_NAMES)
    out_by_scene: dict[str, dict[str, list[float]]] = {}
    n_inst = n_fallback = 0
    for sid, vals in per_scene.items():
        table = class_table.get(sid, {})
        row: dict[str, list[float]] = {}
        for inst_id in vals:
            entry = table.get(inst_id)
            mean = class_mean.get(entry["class"]) if entry else None
            if mean is None:
                mean = zero
                n_fallback += 1
            row[inst_id] = mean
            n_inst += 1
        out_by_scene[sid] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "by_scene": out_by_scene,
                "meta": {
                    "split": str(args.split),
                    "class_table": str(args.class_table),
                    "n_train_scenes": len(train_ids),
                    "n_classes_fitted": len(class_mean),
                    "n_instances": n_inst,
                    "n_fallback_to_global_mean": n_fallback,
                    "n_instances_without_class": n_no_class,
                    "property_names": list(PROPERTY_NAMES),
                },
            }
        )
    )
    print(
        f"{len(out_by_scene)} scenes, {n_inst} instances, {len(class_mean)} classes fitted "
        f"on {len(train_ids)} train scenes; {n_fallback} fell back to the global mean "
        f"({n_no_class} instances had no class entry) -> {args.out}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit log10 z-score normalization stats for the VLM physics labels.

Applies exactly the transform ``InstasceneVlmPhysGMParser`` uses
(``si_scale`` → optional ``log10(max(v, 1.0))``) and reports the mean/std of the
resulting model-space scalars, so the fitted constants are drop-in replacements
for ``PHYSGM_NORMALIZATION`` in ``src/dataset/physics/parsers.py``.

Fit on the TRAINING split only: pass ``--scene_ids`` with one scene_id per line.
Without it the fit covers every scene in the manifest (report-only).

Example:
  python scripts/fit_physgm_norm.py \
    --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K \
    --manifest /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/manifest_infinigen_phys.jsonl \
    --out /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/physgm_norm_infinigen.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dataset.physics.parsers import (  # noqa: E402
    PHYSGM_NORMALIZATION,
    _VLM_PROP_KEYS,
)
from src.dataset.physics.types import PROPERTY_NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene_ids", type=Path, default=None,
                    help="Training-split scene_ids, one per line. Omit = all scenes.")
    args = ap.parse_args()

    keep: set[str] | None = None
    if args.scene_ids is not None:
        keep = {ln.strip() for ln in args.scene_ids.read_text().splitlines() if ln.strip()}

    cols: list[list[float]] = [[] for _ in PROPERTY_NAMES]
    n_scenes = n_inst = n_skipped = 0
    clamped = [0] * len(PROPERTY_NAMES)

    with args.manifest.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scene = json.loads(line)
            if keep is not None and scene.get("scene_id") not in keep:
                continue
            rel = scene.get("physics_labels_path")
            if rel is None:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = args.root / path
            if not path.is_file():
                continue
            n_scenes += 1
            for r in json.load(path.open("r")):
                if not isinstance(r, dict) or int(r.get("id", 0)) <= 0:
                    continue
                props = (r.get("response") or {}).get("physical_property")
                if not isinstance(props, dict):
                    continue
                try:
                    raw = [float(props[k]["mean"]) for k in _VLM_PROP_KEYS]
                except (KeyError, TypeError, ValueError):
                    n_skipped += 1
                    continue
                n_inst += 1
                for i, (v, spec) in enumerate(zip(raw, PHYSGM_NORMALIZATION, strict=True)):
                    v *= spec.si_scale
                    if spec.log10:
                        if v < 1.0:
                            clamped[i] += 1
                        v = math.log10(max(v, 1.0))
                    cols[i].append(v)

    fitted = {}
    print(f"scenes={n_scenes} instances={n_inst} skipped_malformed={n_skipped}\n")
    print(f"{'property':<16} {'n':>7} {'clamped':>8} "
          f"{'fit_mean':>10} {'fit_std':>9} {'cur_mean':>10} {'cur_std':>9}")
    for i, name in enumerate(PROPERTY_NAMES):
        xs = cols[i]
        mean = statistics.fmean(xs)
        std = statistics.pstdev(xs)
        cur = PHYSGM_NORMALIZATION[i]
        fitted[name] = {
            "si_scale": cur.si_scale,
            "log10": cur.log10,
            "mean": mean,
            "std": std,
            "n": len(xs),
            "clamped_below_1": clamped[i],
        }
        print(f"{name:<16} {len(xs):>7} {clamped[i]:>8} "
              f"{mean:>10.6f} {std:>9.6f} {cur.mean:>10.6f} {cur.std:>9.6f}")

    doc = {
        "manifest": str(args.manifest),
        "scene_ids": str(args.scene_ids) if args.scene_ids else "ALL (report-only fit)",
        "n_scenes": n_scenes,
        "n_instances": n_inst,
        "properties": fitted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()

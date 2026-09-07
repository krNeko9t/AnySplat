"""Split a manifest into train / val **scene** lists.

Why scenes and not classes: 213 Infinigen classes contain 126 singletons
(ticket 04), so a class-held-out split is impossible.  Why *groups* and not
plain scenes: Infinigen scene ids are ``infinigen_<scene_XXX>_<hash>`` and
``scene_XXX`` is a generation shard; splitting whole shards costs nothing and
removes any doubt about shared assets.  (Measured 2026-09-07: object-name
Jaccard within a shard 0.049 vs across shards 0.047 -- no sharing.)

Output is a single JSON holding *both* lists, so train and val can never drift
apart:

    {"val_scene_ids": [...], "train_scene_ids": [...], "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def group_of(scene_id: str, group_by: str) -> str:
    if group_by == "infinigen_shard":
        # infinigen_scene_042_1eac55ca -> infinigen_scene_042
        return scene_id.rsplit("_", 1)[0]
    if group_by == "none":
        return scene_id
    raise ValueError(f"unknown group_by={group_by!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--group_by", default="infinigen_shard", choices=["infinigen_shard", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min_views",
        type=int,
        default=13,
        help=(
            "drop scenes the view sampler would drop anyway, so the split counts are "
            "honest.  Default 13 = ViewSamplerBoundedFixed.min_frames_required for "
            "num_context_views=4 / min_gap_multiplier=3 -- NOT 4.  Recompute it if "
            "either changes."
        ),
    )
    args = ap.parse_args()

    scenes: list[str] = []
    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if len(d.get("frames") or []) < args.min_views:
                continue
            scenes.append(str(d["scene_id"]))

    groups: dict[str, list[str]] = defaultdict(list)
    for sid in scenes:
        groups[group_of(sid, args.group_by)].append(sid)

    target = args.val_fraction * len(scenes)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)

    # Greedy: take shards until the next one would overshoot the target further
    # than stopping short of it.  Deterministic given the seed.
    val_groups: list[str] = []
    n_val = 0
    for k in keys:
        size = len(groups[k])
        if n_val and abs(n_val + size - target) > abs(n_val - target):
            continue
        if n_val >= target:
            break
        val_groups.append(k)
        n_val += size

    val = sorted(s for k in val_groups for s in groups[k])
    val_set = set(val)
    train = sorted(s for s in scenes if s not in val_set)

    out = {
        "val_scene_ids": val,
        "train_scene_ids": train,
        "meta": {
            "manifest": str(args.manifest),
            "group_by": args.group_by,
            "seed": args.seed,
            "val_fraction_requested": args.val_fraction,
            "val_fraction_actual": len(val) / max(1, len(scenes)),
            "n_scenes": len(scenes),
            "n_val_groups": len(val_groups),
            "min_views": args.min_views,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(
        f"{len(scenes)} scenes in {len(groups)} groups -> "
        f"train {len(train)} / val {len(val)} "
        f"({out['meta']['val_fraction_actual']:.3%}, {len(val_groups)} groups) -> {args.out}"
    )


if __name__ == "__main__":
    main()

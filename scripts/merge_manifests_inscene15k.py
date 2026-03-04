#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_manifest_lines(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with path.open("r") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "scenes" in obj:
            scenes = obj["scenes"]
        else:
            scenes = obj
        if not isinstance(scenes, list):
            raise ValueError(f"Manifest {path} must be list or {{'scenes': [...]}}")
        for s in scenes:
            yield s


def add_prefix(scene: dict, prefix: str) -> dict:
    """Optionally prefix scene_id to避免不同子集间冲突."""
    s = dict(scene)
    sid = str(s.get("scene_id", ""))
    s["scene_id"] = f"{prefix}{sid}" if sid else prefix.rstrip("_")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest_infinigen",
        type=Path,
        required=True,
        help="manifest for InsScene-processed_infinigen (e.g. manifest_infinigen.jsonl)",
    )
    ap.add_argument(
        "--manifest_scannetpp_v2",
        type=Path,
        required=True,
        help="manifest for InsScene-processed_scannetpp_v2 (e.g. manifest_scannetpp_v2.jsonl)",
    )
    ap.add_argument(
        "--manifest_re10k",
        type=Path,
        required=True,
        help="manifest for InsScene-processed_re10k (e.g. manifest_re10k.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output merged manifest .jsonl (e.g. manifest_inscene15k.jsonl)",
    )
    ap.add_argument(
        "--prefixes",
        type=str,
        default="inf_,spp_,re10k_",
        help="Comma-separated scene_id prefixes for (infinigen, scannetpp_v2, re10k).",
    )
    args = ap.parse_args()

    prefixes = args.prefixes.split(",")
    if len(prefixes) != 3:
        raise ValueError("prefixes must have exactly 3 entries")
    p_inf, p_spp, p_re = prefixes

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_scenes = 0
    with out_path.open("w") as f:
        for scene in iter_manifest_lines(args.manifest_infinigen):
            f.write(json.dumps(add_prefix(scene, p_inf)) + "\n")
            num_scenes += 1
        for scene in iter_manifest_lines(args.manifest_scannetpp_v2):
            f.write(json.dumps(add_prefix(scene, p_spp)) + "\n")
            num_scenes += 1
        for scene in iter_manifest_lines(args.manifest_re10k):
            f.write(json.dumps(add_prefix(scene, p_re)) + "\n")
            num_scenes += 1

    print(f"Wrote {num_scenes} scenes to {out_path}")


if __name__ == "__main__":
    main()


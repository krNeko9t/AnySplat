#!/usr/bin/env python3
"""Smoke-check the Infinigen physics manifest end to end.

Loads real samples through ``DatasetManifest`` with the
``instascene_vlm_physgm`` parser, denormalizes the targets back to SI, joins the
Infinigen class table, and prints ``(class, E, nu, rho)`` per instance — the
completion criterion for ticket 02.

Example:
  python scripts/check_infinigen_phys_batch.py \
    --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K \
    --manifest manifest_infinigen_phys.jsonl \
    --class_table /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/infinigen_class_table.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dataset.dataset_manifest import DatasetManifest, DatasetManifestCfg  # noqa: E402
from src.dataset.physics.parsers import physgm_denormalize  # noqa: E402
from src.dataset.view_sampler.view_sampler_bounded_fixed import (  # noqa: E402
    ViewSamplerBoundedFixed,
    ViewSamplerBoundedFixedCfg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--class_table", type=Path, default=None)
    ap.add_argument("--num_scenes", type=int, default=3)
    ap.add_argument("--num_instances", type=int, default=12)
    ap.add_argument("--num_context_views", type=int, default=4)
    args = ap.parse_args()

    sampler_cfg = ViewSamplerBoundedFixedCfg(
        name="bounded_fixed",
        num_context_views=args.num_context_views,
        num_target_views=0,
        min_distance_between_context_views=12,
        max_distance_between_context_views=24,
        min_distance_to_context_views=0,
        warm_up_steps=0,
        initial_min_distance_between_context_views=2,
        initial_max_distance_between_context_views=6,
        max_img_per_gpu=24,
        min_gap_multiplier=3,
        max_gap_multiplier=5,
    )
    cfg = DatasetManifestCfg(
        original_image_shape=[288, 512],
        input_image_shape=[224, 392],
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=sampler_cfg,
        name="manifest",
        root=args.root,
        manifest_path=args.manifest,
        physics_parser="instascene_vlm_physgm",
    )
    ds = DatasetManifest(cfg, "train", ViewSamplerBoundedFixed(sampler_cfg, "train", False, False, None))
    print(f"scenes in manifest: {len(ds)}")

    class_table = json.loads(args.class_table.read_text()) if args.class_table else {}

    for i in range(args.num_scenes):
        ex = ds.getitem(i, args.num_context_views, (16, 28))
        scene_id = ds.scenes[i]["scene_id"]
        tgt = ex.get("physgm_target")
        img = ex["context"]["image"]
        mask = ex["context"]["instance_mask"]
        print(f"\n=== {scene_id}  image={tuple(img.shape)} mask={tuple(mask.shape)} ===")
        if tgt is None:
            print("  NO physgm_target")
            continue
        ids = torch.nonzero(tgt.valid, as_tuple=False).flatten()
        si = physgm_denormalize(tgt.value_lut[ids])  # density kg/m³, E Pa, nu
        in_mask = set(int(v) for v in torch.unique(mask))
        print(f"  labelled instances={len(ids)}  ids also visible in sampled masks="
              f"{len(in_mask & set(int(v) for v in ids))}")
        names = class_table.get(scene_id, {})
        print(f"  {'id':>5} {'class':<22} {'E (MPa)':>12} {'nu':>7} {'rho (kg/m3)':>12}")
        for inst in ids[: args.num_instances]:
            k = int(inst)
            row = si[(ids == inst).nonzero()[0, 0]]
            cls = names.get(str(k), {}).get("class", "?")
            print(f"  {k:>5} {cls:<22} {row[1].item() / 1e6:>12.4g} "
                  f"{row[2].item():>7.3f} {row[0].item():>12.4g}")


if __name__ == "__main__":
    main()

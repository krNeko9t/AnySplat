"""Dump PhysGM z-target statistics over a manifest.

Answers two diagnostic questions for scheme physgm_copy:
  1. What is Var(z) per property? A model that predicts the dataset mean
     converges to physgm_<prop>_mse == Var(z); only MSE *below* that means
     the readout is actually extracting signal from the features.
  2. Are there outlier labels (|z| > 3)? Each one contributes z^2 to MSE and
     shows up as a loss spike whenever its scene is sampled.

Usage (on the training machine):
  python scripts/physgm_target_stats.py \
      --manifest /home/liaoyuanjun/projects/AnySplat/manifests/manifest_phys_scannet100.jsonl \
      --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/processed_scannetpp_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.dataset.physics.parsers import InstasceneVlmPhysGMParser
from src.dataset.physics.types import PROPERTY_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--outlier-z", type=float, default=3.0)
    args = ap.parse_args()

    parser = InstasceneVlmPhysGMParser()
    zs: list[torch.Tensor] = []
    n_scenes = 0
    n_scenes_with_labels = 0

    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scene = json.loads(line)
            n_scenes += 1
            target = parser.parse(scene, args.root)
            if target is None:
                continue
            n_scenes_with_labels += 1
            zs.append(target.value_lut[target.valid])

    if not zs:
        print(f"no labels parsed from {n_scenes} scenes")
        return

    z = torch.cat(zs, dim=0)  # [N, P]
    print(f"scenes: {n_scenes} ({n_scenes_with_labels} with labels), instances: {z.shape[0]}")
    print(f"{'property':<16} {'mean':>8} {'std':>8} {'var':>8} {'min':>8} {'max':>8} {'|z|>' + str(args.outlier_z):>8}")
    for i, name in enumerate(PROPERTY_NAMES):
        col = z[:, i]
        n_out = int((col.abs() > args.outlier_z).sum())
        print(
            f"{name:<16} {col.mean():>8.3f} {col.std():>8.3f} {col.var():>8.3f}"
            f" {col.min():>8.3f} {col.max():>8.3f} {n_out:>8d}"
        )
    print(
        "\nbaseline: physgm_<prop>_mse stuck at ~var above means the model only"
        " learned the dataset mean; MSE below var means real signal."
    )


if __name__ == "__main__":
    main()

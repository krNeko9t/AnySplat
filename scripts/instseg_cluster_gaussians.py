# deprecated: 用 instseg_infer.py 替代，此脚本仅保留作为参考
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from src.instseg.kmeans import kmeans_torch


def make_color_map(k: int) -> np.ndarray:
    # Simple deterministic palette.
    rng = np.random.default_rng(0)
    colors = rng.integers(low=0, high=255, size=(k, 3), dtype=np.uint8)
    return colors


def export_colored_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    assert xyz.shape[0] == rgb.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elems = np.empty(xyz.shape[0], dtype=dtype)
    elems["x"] = xyz[:, 0]
    elems["y"] = xyz[:, 1]
    elems["z"] = xyz[:, 2]
    elems["red"] = rgb[:, 0]
    elems["green"] = rgb[:, 1]
    elems["blue"] = rgb[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elems, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_pt", type=str, required=True, help="Path to exported embedding .pt")
    ap.add_argument("--k", type=int, required=True, help="Number of instances/clusters")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/instseg")
    args = ap.parse_args()

    emb_path = Path(args.embedding_pt)
    payload = torch.load(emb_path, map_location="cpu")
    means = payload["means"].float().numpy()
    feats = payload["instance_feat"].float()

    labels, centers = kmeans_torch(feats, k=args.k, num_iters=args.iters, seed=args.seed)
    labels_np = labels.cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels_np)
    torch.save({"centers": centers.cpu()}, out_dir / "centers.pt")

    colors = make_color_map(args.k)
    rgb = colors[labels_np]
    export_colored_points_ply(out_dir / "gaussians_instances.ply", means, rgb)
    print(f"Saved labels to {out_dir/'labels.npy'} and colored ply to {out_dir/'gaussians_instances.ply'}")


if __name__ == "__main__":
    main()


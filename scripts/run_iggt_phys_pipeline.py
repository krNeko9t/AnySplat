#!/usr/bin/env python3
"""One command: a pre-built 3DGS scene + its cameras -> 3D instances carrying (E, nu, rho).

Ticket 05 of the ``iggt-phys-pipeline`` map -- the reproducible script behind the
delivered figures and the instance physics table.

What it runs, in order
----------------------
1. **trace** -- ``trace_instance_to_gaussians.py --feature_source iggt_phys``:
   ONE IGGT forward pass over the scene's own views produces both the 8-d
   instance feature map and the 32-d ``physgm_dpt`` dense physics feature map;
   three chunked rasterizer passes (TRACE_CHANNELS=20) carry them onto the same
   Gaussians, enforced by a bit-exact ``num_ray`` assertion per pass.
2. **cluster** -- k=20 3D-KNN feature smoothing, L2 normalise, then
   ``src.instseg.hdbscan_assign`` at the shipped operating point
   (50 / 10 / 0.06, ``max_points=200000``; ticket 03 showed this is the only
   grid point that does not degrade on two scenes -- nothing here is tuned).
3. **physics** -- for each instance, pool its Gaussians' physics features and
   push the single pooled 32-vector through the *trained* per-property MLP,
   then ``physgm_denormalize`` last.  Pooling is literally
   ``sum_g gau_sem_g = sum_g num_ray_g * gau_phys_feat_g``; the denominator is
   irrelevant because the decoder's first layer is a LayerNorm, which is
   invariant to positive scaling (ticket 01's correction block).
4. **overlay** -- ``render_instance_overlay.py``: the original photograph with
   semi-transparent instance colours composited on top, at native resolution.
   Deliberately NOT a grey-background 3DGS render.

Two columns that must never be merged (ticket 01 (d)):
  * ``var``       = the head's own output, ``softplus(out[...,1]) + 1e-2`` -- the
                    model's uncertainty about this instance;
  * ``mu_spread`` = the standard deviation of ``mu`` obtained by decoding *each
                    Gaussian of the instance separately* through the same MLP --
                    member consistency.  This only means anything because the
                    LayerNorm makes the decoder scale-invariant: per-Gaussian
                    vectors are ~50x shorter than the pooled one and their
                    per-Gaussian scale spreads over ~100x.

Example
-------
    CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 python scripts/run_iggt_phys_pipeline.py \
        --scene /home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench \
        --ply   /home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench/point_cloud.ply \
        --iggt_model_path /home/liaoyuanjun/iggt_checkpoint.pth \
        --phys_ckpt /mnt/storage_pool/liaoyuanjun/runs/ticket05/physhead_step10000.pt \
        --out_dir /mnt/storage_pool/liaoyuanjun/runs/ticket05/bench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PYTHON = sys.executable
DEFAULT_HDBSCAN = dict(min_cluster_size=50, min_samples=10,
                       cluster_selection_epsilon=0.06, max_points=200_000)


# --------------------------------------------------------------------------- #
# stage 1: trace
# --------------------------------------------------------------------------- #
def run_trace(args, trace_dir: Path) -> Path:
    out_pt = trace_dir / "gaussian_iggt_phys_feat.pt"
    if out_pt.exists() and not args.force_trace:
        print(f"[trace] reusing {out_pt}")
        return out_pt
    cmd = [
        PYTHON, str(REPO / "scripts" / "trace_instance_to_gaussians.py"),
        "--source_path", args.scene,
        "--ply_path", args.ply,
        "--output_dir", str(trace_dir),
        "--feature_source", "iggt_phys",
        "--iggt_model_path", args.iggt_model_path,
        "--iggt_phys_ckpt", args.phys_ckpt,
        "--iggt_phys_image_size", args.image_size,
        "--encoder_batch_size", str(args.encoder_batch_size),
        "--camera_backend", args.camera_backend,
        "--images_folder", args.images_folder,
        "--resolution", str(args.resolution),
        # ticket 03/04 operating point: a full-image resize is a no-op for
        # TraceCamera, so the trace runs on the scene's own cameras.  Keeping
        # this identical is what makes the instance stream comparable with the
        # official-IGGT baseline.
        "--no_trace_crop_aligned",
        "--trace_blend_mass",
    ]
    if args.transforms_json:
        cmd += ["--transforms_json", args.transforms_json,
                "--transforms_axis", args.transforms_axis]
    if args.max_views:
        cmd += ["--max_views", str(args.max_views)]
    print("[trace] " + " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(REPO))
    print(f"[trace] {time.perf_counter() - t0:.1f} s -> {out_pt}")
    return out_pt


# --------------------------------------------------------------------------- #
# stage 2: cluster
# --------------------------------------------------------------------------- #
def run_cluster(traced_pt: Path, ply: str, out_npz: Path, knn_k: int,
                params: dict, force: bool) -> np.ndarray:
    if out_npz.exists() and not force:
        print(f"[cluster] reusing {out_npz}")
        return np.load(out_npz)["labels"].astype(np.int32)

    from scripts.trace_instance_to_gaussians import knn_smooth_gaussians
    from src.instseg.hdbscan_assign import hdbscan_assign
    from src.trace_render.trace_rasterize import load_gaussians_from_ply

    ck = torch.load(traced_pt, map_location="cpu", weights_only=True)
    feat = ck["feat"]                      # 8-d instance feature, per Gaussian
    valid = (ck["num_ray"] > 0).numpy()
    print(f"[cluster] {int(valid.sum()):,}/{valid.shape[0]:,} traced "
          f"({100 * valid.mean():.2f}%)")

    if knn_k > 0:
        means, *_ = load_gaussians_from_ply(ply)
        xyz = means.detach().cpu().numpy()
        t0 = time.perf_counter()
        feat = knn_smooth_gaussians(xyz, feat.cuda(), k=knn_k).cpu()
        print(f"[cluster] knn smoothing k={knn_k}: {time.perf_counter() - t0:.1f} s")
        del means, xyz

    x = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)[valid].numpy()
    t0 = time.perf_counter()
    lab = hdbscan_assign(x, rng_seed=0, **params)
    print(f"[cluster] hdbscan {params}: {time.perf_counter() - t0:.1f} s")

    labels = np.zeros(valid.shape[0], dtype=np.int32)
    labels[valid] = lab + 1                # 0 = untraced (trace-script convention)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, labels=labels, valid=valid,
                        params=json.dumps(params))
    k = int(labels.max())
    sizes = np.bincount(labels, minlength=k + 1)[1:]
    print(f"[cluster] {k} instances, sizes {sorted(sizes.tolist(), reverse=True)[:12]}"
          f" -> {out_npz}")
    return labels


# --------------------------------------------------------------------------- #
# stage 3: physics
# --------------------------------------------------------------------------- #
def load_property_decoders(phys_ckpt: str, feat_dim: int, hidden: int, device: str):
    """The three trained per-property MLPs, and nothing else from the checkpoint."""
    import torch.nn as nn

    from src.dataset.physics.types import PROPERTY_NAMES
    from src.model.heads.physics.physgm_readout import _make_property_decoder

    raw = torch.load(phys_ckpt, map_location="cpu", weights_only=False)
    step = None
    if isinstance(raw, dict) and "state_dict" in raw:
        step = raw.get("global_step")
        raw = raw["state_dict"]
    raw = {k.replace("module.", "", 1): v for k, v in raw.items()}
    raw = {(k[len("model."):] if k.startswith("model.") else k): v
           for k, v in raw.items()}

    decoders = nn.ModuleList(
        _make_property_decoder(feat_dim, hidden) for _ in PROPERTY_NAMES
    )
    pre = "encoder.physics_scheme.physgm_dense_readout.decoders."
    sd = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
    decoders.load_state_dict(sd, strict=True)  # raises if the head does not match
    decoders.eval().to(device)
    for p in decoders.parameters():
        p.requires_grad = False
    print(f"[physics] loaded {len(sd)} decoder tensors from {phys_ckpt} "
          f"(global_step={step})")
    return decoders, step


@torch.no_grad()
def decode(decoders, pooled: torch.Tensor):
    """``[K, C]`` model-space features -> ``(mu [K,P], var [K,P])``.

    Byte-identical arithmetic to ``PhysGMDenseReadout.forward``'s decode step.
    """
    outs = [d(pooled.float()) for d in decoders]     # P x [K, 2]
    out = torch.stack(outs, dim=1)                   # [K, P, 2]
    return out[..., 0], F.softplus(out[..., 1]) + 1e-2


@torch.no_grad()
def instance_physics(traced_pt: Path, labels: np.ndarray, decoders,
                     device: str, spread_chunk: int = 200_000) -> list[dict]:
    from src.dataset.physics.parsers import physgm_denormalize

    ck = torch.load(traced_pt, map_location="cpu", weights_only=True)
    phys = ck["gau_phys_feat"]                        # [N, 32], = gau_sem / num_ray
    num_ray = ck["num_ray"]                           # [N]
    if phys.shape[0] != labels.shape[0]:
        raise SystemExit(f"labels {labels.shape[0]} vs features {phys.shape[0]}")

    lab_t = torch.from_numpy(labels)
    k = int(labels.max())
    rows = []
    for inst in range(1, k + 1):
        sel = torch.nonzero(lab_t == inst, as_tuple=True)[0]
        n = int(sel.numel())
        f = phys[sel].to(device)                      # [n, 32]
        nr = num_ray[sel].to(device)                  # [n]

        # Pooling == "sum gau_sem over the instance's Gaussians" (ticket 01).
        # gau_sem_g = num_ray_g * gau_phys_feat_g.  Any positive scaling of this
        # is erased by the decoder's first LayerNorm, so no denominator is applied.
        pooled = (f * nr.unsqueeze(-1)).sum(0, keepdim=True)   # [1, 32]
        mu, var = decode(decoders, pooled)                     # [1, P]
        si = physgm_denormalize(mu)[0]                         # rho, E, nu

        # mu_spread: decode each Gaussian on its own through the SAME MLP.
        acc_sum = torch.zeros(3, dtype=torch.float64, device=device)
        acc_sq = torch.zeros(3, dtype=torch.float64, device=device)
        for s in range(0, n, spread_chunk):
            mu_g, _ = decode(decoders, f[s:s + spread_chunk])
            acc_sum += mu_g.double().sum(0)
            acc_sq += (mu_g.double() ** 2).sum(0)
        mean_g = acc_sum / max(n, 1)
        spread = torch.sqrt(torch.clamp(acc_sq / max(n, 1) - mean_g ** 2, min=0.0))

        rows.append({
            "instance_id": inst,
            "n_gau": n,
            "n_ray_total": float(nr.sum()),
            "rho_si": float(si[0]), "E_si": float(si[1]), "nu_si": float(si[2]),
            "mu_z": [float(v) for v in mu[0]],
            "var": [float(v) for v in var[0]],
            "mu_spread": [float(v) for v in spread],
            "mu_mean_per_gaussian_z": [float(v) for v in mean_g],
        })
        del f, nr
    return rows


# --------------------------------------------------------------------------- #
# stage 4: overlay
# --------------------------------------------------------------------------- #
def run_overlay(args, labels_npz: Path, out_dir: Path) -> Path:
    cmd = [
        PYTHON, str(REPO / "scripts" / "render_instance_overlay.py"),
        "--source_path", args.scene, "--ply", args.ply,
        "--labels", str(labels_npz), "--out_dir", str(out_dir),
        "--camera_backend", args.camera_backend,
        "--images_folder", args.images_folder,
        "--resolution", str(args.resolution),
        "--n_views", str(args.n_overlay_views),
    ]
    if args.overlay_views:
        cmd += ["--view_names", args.overlay_views]
    if args.transforms_json:
        cmd += ["--transforms_json", args.transforms_json,
                "--transforms_axis", args.transforms_axis]
    print("[overlay] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO))
    return out_dir / "contact_sheet.png"


# --------------------------------------------------------------------------- #
def write_table(rows, path_stem: Path, meta: dict, small_threshold: int,
                names: dict | None = None):
    names = names or {}
    props = ["rho", "E", "nu"]
    cols = (["instance_id", "name", "n_gau", "trust", "rho_si_kg_m3", "E_si_Pa", "nu_si"]
            + [f"var_{n}" for n in props] + [f"mu_spread_{n}" for n in props])
    lines = [",".join(cols)]
    for r in rows:
        trust = "LOW(small)" if r["n_gau"] < small_threshold else "ok"
        nm = names.get(str(r["instance_id"]), "").replace(",", ";")
        lines.append(",".join([
            str(r["instance_id"]), nm, str(r["n_gau"]), trust,
            f"{r['rho_si']:.6g}", f"{r['E_si']:.6g}", f"{r['nu_si']:.6g}",
            *[f"{v:.6g}" for v in r["var"]],
            *[f"{v:.6g}" for v in r["mu_spread"]],
        ]))
    path_stem.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["| instance_id | name | n_gau | trust | rho (kg/m3) | E (Pa) | nu | "
          "var_rho | var_E | var_nu | spread_rho | spread_E | spread_nu |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        trust = "**LOW (small)**" if r["n_gau"] < small_threshold else "ok"
        md.append("| {id} | {nm} | {n:,} | {t} | {rho:.1f} | {E:.3e} | {nu:.4f} | "
                  "{v0:.4f} | {v1:.4f} | {v2:.4f} | {s0:.4f} | {s1:.4f} | {s2:.4f} |".format(
                      id=r["instance_id"], nm=names.get(str(r["instance_id"]), "-"),
                      n=r["n_gau"], t=trust,
                      rho=r["rho_si"], E=r["E_si"], nu=r["nu_si"],
                      v0=r["var"][0], v1=r["var"][1], v2=r["var"][2],
                      s0=r["mu_spread"][0], s1=r["mu_spread"][1], s2=r["mu_spread"][2]))
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with open(path_stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("\n".join(md))
    print(f"[table] -> {path_stem}.{{csv,md,json}}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene source_path (has images/, sparse/)")
    ap.add_argument("--ply", required=True, help="the pre-built 3DGS point_cloud.ply")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iggt_model_path", default="/home/liaoyuanjun/iggt_checkpoint.pth")
    ap.add_argument("--phys_ckpt", required=True,
                    help="trained physgm_dpt head (Lightning .ckpt or extracted .pt)")
    ap.add_argument("--image_size", default="504,336",
                    help="W,H fed to IGGT (default: the iggt operating point, so the "
                         "instance stream matches the official-IGGT baseline)")
    ap.add_argument("--encoder_batch_size", type=int, default=48,
                    help="views per IGGT forward.  IGGT's instance feature space is "
                         "batch-relative (map G8), so a scene that fits in one batch "
                         "should get one batch")
    ap.add_argument("--camera_backend", default="colmap")
    ap.add_argument("--images_folder", default="images")
    ap.add_argument("--resolution", type=int, default=1)
    ap.add_argument("--transforms_json", default=None)
    ap.add_argument("--transforms_axis", default="opencv")
    ap.add_argument("--max_views", type=int, default=None)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--min_cluster_size", type=int, default=DEFAULT_HDBSCAN["min_cluster_size"])
    ap.add_argument("--min_samples", type=int, default=DEFAULT_HDBSCAN["min_samples"])
    ap.add_argument("--cluster_selection_epsilon", type=float,
                    default=DEFAULT_HDBSCAN["cluster_selection_epsilon"])
    ap.add_argument("--max_points", type=int, default=DEFAULT_HDBSCAN["max_points"])
    ap.add_argument("--small_instance_threshold", type=int, default=2000,
                    help="rows below this many Gaussians are flagged LOW trust "
                         "(ticket 04: 2D<->3D pooling cosine collapses to 0.39-0.82 there)")
    ap.add_argument("--n_overlay_views", type=int, default=6)
    ap.add_argument("--overlay_views", default=None, help="comma-separated view names")
    ap.add_argument("--labels", default=None,
                    help="reuse an existing labels .npz instead of clustering")
    ap.add_argument("--keep_phys_feat", action="store_true", default=True,
                    help="write a compact fp16 copy of gau_phys_feat so a later head "
                         "or threshold change can skip the 3-pass trace")
    ap.add_argument("--no_keep_phys_feat", dest="keep_phys_feat", action="store_false")
    ap.add_argument("--names_json", default=None,
                    help='optional {"1": "cat figurine", "8": "wall  [stuff]", ...}: '
                         "puts a human-readable name (and the stuff marking) in the table")
    ap.add_argument("--force_trace", action="store_true")
    ap.add_argument("--force_cluster", action="store_true")
    ap.add_argument("--skip_overlay", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()

    traced = run_trace(args, out / "trace")

    params = dict(min_cluster_size=args.min_cluster_size,
                  min_samples=args.min_samples,
                  cluster_selection_epsilon=args.cluster_selection_epsilon,
                  max_points=args.max_points)
    if args.labels:
        labels_npz = Path(args.labels)
        labels = np.load(labels_npz)["labels"].astype(np.int32)
        print(f"[cluster] using supplied labels {labels_npz} "
              f"({int(labels.max())} instances)")
    else:
        labels_npz = out / "labels.npz"
        labels = run_cluster(traced, args.ply, labels_npz, args.knn_k, params,
                             args.force_cluster)

    decoders, step = load_property_decoders(args.phys_ckpt, 32, 64, device)
    rows = instance_physics(traced, labels, decoders, device)
    meta = {
        "scene": os.path.abspath(args.scene),
        "ply": os.path.abspath(args.ply),
        "traced_pt": str(traced),
        "labels_npz": str(labels_npz),
        "image_size": args.image_size,
        "encoder_batch_size": args.encoder_batch_size,
        "hdbscan": params,
        "knn_k": args.knn_k,
        "phys_ckpt": os.path.abspath(args.phys_ckpt),
        "phys_ckpt_global_step": step,
        "iggt_ckpt": args.iggt_model_path,
        "property_order": ["density_kg_m3", "youngs_modulus_Pa", "poisson_ratio"],
        "small_instance_threshold": args.small_instance_threshold,
        "pooling": "sum_g num_ray_g * gau_phys_feat_g  (== sum_g gau_sem_g); "
                   "no denominator -- the decoder's first LayerNorm is scale-invariant",
    }
    names = json.loads(Path(args.names_json).read_text()) if args.names_json else {}
    meta["names"] = names
    write_table(rows, out / "instance_physics", meta, args.small_instance_threshold, names)

    if args.keep_phys_feat:
        ck = torch.load(traced, map_location="cpu", weights_only=True)
        torch.save({"gau_phys_feat_fp16": ck["gau_phys_feat"].half(),
                    "num_ray": ck["num_ray"],
                    "gaussian_index": ck["gaussian_index"],
                    "provenance": ck.get("provenance")},
                   out / "gau_phys_feat_fp16.pt")
        print(f"[physics] cached fp16 features -> {out / 'gau_phys_feat_fp16.pt'}")

    if not args.skip_overlay:
        sheet = run_overlay(args, labels_npz, out / "overlay")
        print(f"[overlay] -> {sheet}")

    print(f"\n[done] {time.perf_counter() - t_start:.1f} s total -> {out}")


if __name__ == "__main__":
    main()

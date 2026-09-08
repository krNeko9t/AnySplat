"""Measure the cost and parameter sensitivity of ``src/instseg/hdbscan_assign``.

The shared module (``subsample -> HDBSCAN -> NearestCentroid``) has never been
timed. This script reproduces it stage by stage with wall-clock and peak-RSS
instrumentation, then sweeps ``max_points`` and the three HDBSCAN parameters.

It reads a ``gaussian_*_feat.pt`` produced by
``scripts/trace_instance_to_gaussians.py`` and never writes into the trace
script's own output tree.

Modes
-----
``cost``       one run at the baseline setting, staged timings + peak RSS
``max_points`` sweep the subsample cap, report instance count + label drift
``params``     perturb min_cluster_size / min_samples / cluster_selection_epsilon
``control``    run the baseline twice to establish the noise floor

Usage
-----
    python scripts/hdbscan_cost_sweep.py --traced run/gaussian_iggt_feat.pt \
        --ply scene/point_cloud.ply --knn_k 20 --modes cost,max_points,params \
        --out_json out/hdbscan_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    with open("/proc/self/statm", encoding="utf-8") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


class RssSampler:
    """Poll RSS in a thread; ``peak`` is the max seen while running."""

    def __init__(self, hz: float = 50.0):
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self.peak = 0.0
        self.start_rss = 0.0
        self._t: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            self._stop.wait(self._interval)

    def __enter__(self):
        self.start_rss = _rss_mb()
        self.peak = self.start_rss
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())
        return False


def hdbscan_assign_timed(
    features: np.ndarray,
    *,
    cluster_selection_epsilon: float = 0.06,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    max_points: int = 200_000,
    rng_seed: int = 0,
):
    """Line-for-line replica of ``src.instseg.hdbscan_assign.hdbscan_assign``
    with per-stage wall clock and peak RSS. Verified label-identical in
    ``--modes control``."""
    import hdbscan as hdb_lib
    from sklearn.neighbors import NearestCentroid

    stats: dict = {}
    N = features.shape[0]
    S = min(max_points, N)
    rng = np.random.default_rng(rng_seed)

    # --- stage 1: subsample -------------------------------------------------
    with RssSampler() as m:
        t0 = time.perf_counter()
        if S < N:
            idx_sub = rng.choice(N, S, replace=False)
            sub_feat = features[idx_sub]
        else:
            sub_feat = features
        t1 = time.perf_counter()
    stats["subsample_s"] = t1 - t0
    stats["subsample_peak_rss_mb"] = m.peak
    stats["n_total"] = int(N)
    stats["n_subsample"] = int(sub_feat.shape[0])

    # --- stage 2: HDBSCAN ---------------------------------------------------
    with RssSampler() as m:
        t0 = time.perf_counter()
        sub_labels = hdb_lib.HDBSCAN(
            cluster_selection_epsilon=cluster_selection_epsilon,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        ).fit_predict(sub_feat)
        t1 = time.perf_counter()
    stats["hdbscan_s"] = t1 - t0
    stats["hdbscan_peak_rss_mb"] = m.peak
    stats["hdbscan_rss_delta_mb"] = m.peak - m.start_rss

    mask_clustered = sub_labels >= 0
    n_clusters = int(sub_labels.max() + 1) if mask_clustered.any() else 0
    stats["n_clusters_subsample"] = n_clusters
    stats["n_noise_subsample"] = int((~mask_clustered).sum())
    stats["noise_frac_subsample"] = float((~mask_clustered).mean())

    # --- stage 3: NearestCentroid full assignment ---------------------------
    with RssSampler() as m:
        t0 = time.perf_counter()
        if n_clusters >= 2:
            nc = NearestCentroid()
            nc.fit(sub_feat[mask_clustered], sub_labels[mask_clustered])
            labels = nc.predict(features)
        else:
            labels = np.zeros(N, dtype=np.int32)
        t1 = time.perf_counter()
    stats["nearest_centroid_s"] = t1 - t0
    stats["nearest_centroid_peak_rss_mb"] = m.peak
    stats["nearest_centroid_rss_delta_mb"] = m.peak - m.start_rss

    # --- stage 4: contiguous relabel ---------------------------------------
    with RssSampler() as m:
        t0 = time.perf_counter()
        unique = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.vectorize(remap.get)(labels).astype(np.int32)
        t1 = time.perf_counter()
    stats["relabel_s"] = t1 - t0
    stats["relabel_peak_rss_mb"] = m.peak

    stats["total_s"] = (
        stats["subsample_s"] + stats["hdbscan_s"]
        + stats["nearest_centroid_s"] + stats["relabel_s"]
    )
    stats["peak_rss_mb"] = max(
        stats["subsample_peak_rss_mb"], stats["hdbscan_peak_rss_mb"],
        stats["nearest_centroid_peak_rss_mb"], stats["relabel_peak_rss_mb"],
    )
    stats["n_instances_final"] = int(len(unique))
    return labels, stats


# ---------------------------------------------------------------------------
# Labeling comparison
# ---------------------------------------------------------------------------

def compare_labelings(a: np.ndarray, b: np.ndarray, sub_n: int = 300_000,
                      seed: int = 12345) -> dict:
    """ARI + optimal-matching disagreement between two labelings of the same points."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from sklearn.metrics import adjusted_rand_score

    n = a.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sub_n, n), replace=False)
    aa, bb = a[idx], b[idx]

    ari = float(adjusted_rand_score(aa, bb))

    ka, kb = int(aa.max()) + 1, int(bb.max()) + 1
    cont = coo_matrix(
        (np.ones(aa.shape[0]), (aa, bb)), shape=(ka, kb)
    ).toarray()
    r, c = linear_sum_assignment(-cont)
    matched = cont[r, c].sum()
    return {
        "ari": ari,
        "matched_frac": float(matched / aa.shape[0]),
        "disagree_frac": float(1.0 - matched / aa.shape[0]),
        "k_a": ka, "k_b": kb,
    }


def size_profile(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    counts = np.sort(counts)[::-1]
    n = counts.sum()
    return {
        "k": int(counts.size),
        "sizes_top20": [int(x) for x in counts[:20]],
        "largest_frac": float(counts[0] / n),
        "top3_frac": float(counts[:3].sum() / n),
        "n_lt_0p1pct": int((counts < 0.001 * n).sum()),
        "n_lt_1k": int((counts < 1000).sum()),
        "n_lt_100": int((counts < 100).sum()),
    }


# ---------------------------------------------------------------------------
# Feature loading (mirrors run_postprocess in trace_instance_to_gaussians.py)
# ---------------------------------------------------------------------------

def load_xyz_from_ply(ply_path):
    """Copy of ``trace_instance_to_gaussians.load_xyz_from_ply`` (scripts/ is not a package)."""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    return xyz.astype(np.float32)


@torch.no_grad()
def knn_smooth_gaussians(xyz_np, feat, k=20):
    """Copy of ``trace_instance_to_gaussians.knn_smooth_gaussians``."""
    from scipy.spatial import cKDTree

    feat_np = feat.cpu().float().numpy()
    tree = cKDTree(xyz_np)
    _, indices = tree.query(xyz_np, k=k + 1)
    indices = indices[:, 1:]
    smoothed = feat_np[indices].mean(axis=1)
    return torch.from_numpy(smoothed).to(device=feat.device, dtype=feat.dtype)


def load_valid_features(traced_pt: str, ply_path: str | None, knn_k: int,
                        device: str = "cuda"):
    ck = torch.load(traced_pt, map_location="cpu", weights_only=True)
    feat = ck["feat"]
    num_ray = ck["num_ray"]
    valid = num_ray > 0
    timings = {}

    if knn_k > 0:
        p = ply_path or ck.get("ply_path")
        t0 = time.perf_counter()
        xyz = load_xyz_from_ply(p)
        timings["ply_xyz_load_s"] = time.perf_counter() - t0
        with RssSampler() as m:
            t0 = time.perf_counter()
            feat = knn_smooth_gaussians(xyz, feat.to(device), k=knn_k).cpu()
            timings["knn_smooth_s"] = time.perf_counter() - t0
        timings["knn_smooth_peak_rss_mb"] = m.peak

    t0 = time.perf_counter()
    feat_n = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)
    x = feat_n[valid].numpy()
    timings["l2norm_and_gather_s"] = time.perf_counter() - t0
    return x, valid.numpy(), timings, ck


# ---------------------------------------------------------------------------

BASE = dict(min_cluster_size=50, min_samples=10,
            cluster_selection_epsilon=0.06, max_points=200_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traced", required=True)
    ap.add_argument("--ply", default=None)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--modes", default="control,cost,max_points,params")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--labels_npz", default=None,
                    help="Save baseline per-Gaussian labels here (for overlays)")
    ap.add_argument("--min_cluster_size", type=int, default=BASE["min_cluster_size"])
    ap.add_argument("--min_samples", type=int, default=BASE["min_samples"])
    ap.add_argument("--cluster_selection_epsilon", type=float,
                    default=BASE["cluster_selection_epsilon"])
    ap.add_argument("--max_points", type=int, default=BASE["max_points"])
    ap.add_argument("--max_points_grid", default="1000000,500000,200000,100000,50000,20000,10000,5000")
    ap.add_argument("--mcs_grid", default="10,25,50,100,200,400,700,1500")
    ap.add_argument("--ms_grid", default="1,5,10,25,50,100,200")
    ap.add_argument("--eps_grid", default="0.0,0.02,0.04,0.06,0.10,0.15,0.25")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    base = dict(min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
                max_points=args.max_points)

    print(f"[load] {args.traced} (knn_k={args.knn_k})")
    x, valid, load_timings, ck = load_valid_features(args.traced, args.ply, args.knn_k)
    print(f"[load] {x.shape[0]:,} valid / {valid.shape[0]:,} gaussians, D={x.shape[1]}")
    print(f"[load] timings: {load_timings}")

    result = {
        "traced": os.path.abspath(args.traced),
        "ply": args.ply or ck.get("ply_path"),
        "source_path": ck.get("source_path"),
        "feature_source": ck.get("feature_source"),
        "n_gaussians": int(valid.shape[0]),
        "n_valid": int(x.shape[0]),
        "feat_dim": int(x.shape[1]),
        "knn_k": args.knn_k,
        "load_timings": load_timings,
        "baseline_params": base,
    }

    # ---- baseline (always) ----
    print(f"\n[baseline] {base}")
    lab_base, st_base = hdbscan_assign_timed(x, **base)
    st_base["size_profile"] = size_profile(lab_base)
    result["baseline"] = st_base
    print(json.dumps({k: v for k, v in st_base.items() if k != "size_profile"}, indent=1))
    print("  size profile:", st_base["size_profile"])

    if args.labels_npz:
        full = np.zeros(valid.shape[0], dtype=np.int32)
        full[valid] = lab_base + 1  # 0 = untraced, matches trace script convention
        np.savez_compressed(args.labels_npz, labels=full, valid=valid,
                            params=json.dumps(base))
        print(f"  saved labels -> {args.labels_npz}")

    # ---- control: same path twice ----
    if "control" in modes:
        print("\n[control] second identical run (noise floor)")
        from src.instseg.hdbscan_assign import hdbscan_assign as real_fn
        lab_2, st_2 = hdbscan_assign_timed(x, **base)
        lab_real = real_fn(x, **base)
        result["control"] = {
            "repeat_vs_baseline": compare_labelings(lab_base, lab_2),
            "shipped_fn_vs_replica": compare_labelings(lab_base, lab_real),
            "shipped_fn_identical": bool(np.array_equal(lab_base, lab_real)),
            "repeat_identical": bool(np.array_equal(lab_base, lab_2)),
            "repeat_total_s": st_2["total_s"],
            "repeat_peak_rss_mb": st_2["peak_rss_mb"],
        }
        print(json.dumps(result["control"], indent=1))

    # ---- max_points sweep ----
    if "max_points" in modes:
        grid = [int(v) for v in args.max_points_grid.split(",")]
        ref_mp = max(grid)
        print(f"\n[max_points] reference = {ref_mp:,}")
        rows = []
        lab_ref = None
        for mp in sorted(grid, reverse=True):
            kw = dict(base); kw["max_points"] = mp
            lab, st = hdbscan_assign_timed(x, **kw)
            if lab_ref is None:
                lab_ref = lab
            cmp_ref = compare_labelings(lab_ref, lab)
            cmp_base = compare_labelings(lab_base, lab)
            row = {
                "max_points": mp,
                "n_instances": st["n_instances_final"],
                "n_clusters_subsample": st["n_clusters_subsample"],
                "noise_frac_subsample": st["noise_frac_subsample"],
                "hdbscan_s": st["hdbscan_s"],
                "nearest_centroid_s": st["nearest_centroid_s"],
                "total_s": st["total_s"],
                "peak_rss_mb": st["peak_rss_mb"],
                "hdbscan_rss_delta_mb": st["hdbscan_rss_delta_mb"],
                "vs_ref": cmp_ref,
                "vs_default200k": cmp_base,
                "size_profile": size_profile(lab),
            }
            rows.append(row)
            print(f"  mp={mp:>8,}  K={row['n_instances']:>3}  "
                  f"hdb={st['hdbscan_s']:7.2f}s  nc={st['nearest_centroid_s']:6.2f}s  "
                  f"rss={st['peak_rss_mb']:8.0f}MB  ARIvsRef={cmp_ref['ari']:.4f}  "
                  f"disagree={cmp_ref['disagree_frac']*100:5.2f}%")
        result["max_points_sweep"] = rows

    # ---- parameter sensitivity ----
    if "params" in modes:
        result["param_sweep"] = {}
        for pname, grid_s in (("min_cluster_size", args.mcs_grid),
                              ("min_samples", args.ms_grid),
                              ("cluster_selection_epsilon", args.eps_grid)):
            cast = float if pname == "cluster_selection_epsilon" else int
            grid = [cast(v) for v in grid_s.split(",")]
            print(f"\n[params] {pname}: {grid}")
            rows = []
            for v in grid:
                kw = dict(base); kw[pname] = v
                lab, st = hdbscan_assign_timed(x, **kw)
                cmp_base = compare_labelings(lab_base, lab)
                row = {
                    pname: v,
                    "n_instances": st["n_instances_final"],
                    "n_clusters_subsample": st["n_clusters_subsample"],
                    "noise_frac_subsample": st["noise_frac_subsample"],
                    "hdbscan_s": st["hdbscan_s"],
                    "total_s": st["total_s"],
                    "peak_rss_mb": st["peak_rss_mb"],
                    "vs_default": cmp_base,
                    "size_profile": size_profile(lab),
                }
                rows.append(row)
                sp = row["size_profile"]
                print(f"  {pname}={v:<8}  K={row['n_instances']:>3}  "
                      f"noise={row['noise_frac_subsample']*100:5.1f}%  "
                      f"hdb={st['hdbscan_s']:6.2f}s  largest={sp['largest_frac']*100:5.1f}%  "
                      f"tiny(<0.1%)={sp['n_lt_0p1pct']:>3}  ARIvsDef={cmp_base['ari']:.4f}")
            result["param_sweep"][pname] = rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\n[done] -> {args.out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ticket 06 of the iggt-phys-pipeline map: the *membership-set* inequivalence.

Training pools the pixels a GT instance mask covers; inference pools the
Gaussians HDBSCAN clustered.  To compare them one needs a **3D** GT instance
set -- and ticket 03 (map fact G9) showed that ``3dovs/bench``'s ``sam/mask``
is per-frame SAM output whose ids do **not** denote the same object across
frames.  So this script runs in two stages:

  STAGE 1 (``--stage link``): build a 3D GT instance set, or prove it cannot be
  built.  Every ``(frame, id)`` mask is back-projected to the Gaussians on its
  own -- one-hot channels **plus a constant-1.0 coverage plane** through the
  same cameras and the same Gaussians as the traced ``.pt`` (ticket 04's
  machinery, copied from ``check_iggt_phys_trace.py``, not reinvented), then
  ``share = occ / coverage``, ``argmax`` with ``share > 0.5``.  Cross-frame ids
  whose Gaussian sets overlap are linked by connected components over an
  overlap threshold; the threshold sweep and the *contradiction* count (a
  component holding two ids from one and the same frame -- SAM says those are
  different objects) decide whether the linking is usable.

  STAGE 2 (``--stage physics``, only meaningful if stage 1 succeeded): side A =
  the linked 3D GT instances, side B = ticket 03's HDBSCAN labels; both pooled
  from the *same* ``gau_phys_feat`` through the *same* MLP.

Pooling is ticket 01's, as corrected by ticket 04: sum ``gau_sem`` over the
instance's Gaussians (``sum num_ray * feat``); the denominator is irrelevant
because the decoder's first layer is a LayerNorm, invariant to positive scaling.

Nothing is compared with ``torch.equal`` except integer ray counts (map F4).

Reproduce (bench, ~90 s with the cache, ~5 min without; GPU 5):

    # 1. re-trace with the trained head -- gau_phys_feat depends on it, and the
    #    ticket-04 .pt was traced at step 500.  --no_trace_crop_aligned keeps the
    #    geometry bit-identical to ticket 03/04 (num_ray is asserted equal below).
    CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 python \
      scripts/trace_instance_to_gaussians.py \
      -s <scene> -p <scene>/point_cloud.ply -o <out>/bench_step10000 \
      --feature_source iggt_phys --iggt_model_path /home/liaoyuanjun/iggt_checkpoint.pth \
      --iggt_phys_ckpt <run>/checkpoints/epoch_57-step_10000.ckpt \
      --trace_blend_mass --no_trace_crop_aligned --postprocess none
    #    ... and once more into <out>/bench_step10000_run2 for the noise floor.

    # 2. this script
    CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 python \
      scripts/check_iggt_phys_membership.py \
      --traced <out>/bench_step10000/gaussian_iggt_phys_feat.pt \
      --traced_control <out>/bench_step10000_run2/gaussian_iggt_phys_feat.pt \
      --mask_dir <scene>/sam/mask \
      --hdbscan_labels <ticket03>/hdbscan/bench_labels_default.npz \
      --phys_ckpt <run>/checkpoints/epoch_57-step_10000.ckpt \
      --cache <out>/perframe_backproj.pt --tau 0.5 --outdir <out>/report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trace_cameras import load_trace_cameras  # noqa: E402
from src.trace_render.trace_rasterize import (  # noqa: E402
    load_gaussians_from_ply,
    resolve_trace_backend,
    trace_single_view_chunked,
)


# --------------------------------------------------------------------------- io
def load_gt_masks(mask_dir, image_names):
    """One integer instance mask per view, in camera order -> list of [H,W] long."""
    mask_dir = Path(mask_dir)
    out = []
    for name in image_names:
        hit = None
        for ext in (".png", ".PNG", ".npy"):
            p = mask_dir / f"{name}{ext}"
            if p.exists():
                hit = p
                break
        if hit is None:
            raise FileNotFoundError(f"No GT mask for view {name!r} under {mask_dir}")
        arr = np.load(hit) if hit.suffix == ".npy" else np.array(Image.open(hit))
        if arr.ndim != 2:
            raise ValueError(f"{hit}: expected a single-channel id map, got {arr.shape}")
        out.append(torch.from_numpy(arr.astype(np.int64)))
    return out


# ------------------------------------------------------------------ stage 1: link
@torch.no_grad()
def backproject_per_frame(args, device="cuda"):
    """Back-project every ``(frame, id)`` mask separately onto the Gaussians.

    Returns a dict with, per frame, an int16 label map over all N Gaussians
    (-1 = not assigned to any id in this frame) and the frame's visibility mask.
    """
    ck = torch.load(args.traced, map_location="cpu", weights_only=False)
    num_ray = ck["num_ray"]
    valid = ck["valid_mask"]
    N = num_ray.shape[0]

    means, quats, scales, opacities, colors = load_gaussians_from_ply(ck["ply_path"])
    backend = resolve_trace_backend("auto", int(scales.shape[1]))
    cams = load_trace_cameras(ck.get("camera_backend", "colmap"),
                              source_path=ck["source_path"],
                              images_folder=args.images_folder,
                              resolution=1,
                              transforms_json=ck.get("transforms_json"),
                              transforms_axis=ck.get("transforms_axis", "nerfstudio"))
    cams = cams[: ck["n_views"]]
    names = [c.image_name for c in cams]
    masks = load_gt_masks(args.mask_dir, names)

    bg = torch.zeros(3, dtype=torch.float32, device=device)
    frame_labels = torch.full((len(cams), N), -1, dtype=torch.int16)
    frame_vis = torch.zeros((len(cams), N), dtype=torch.bool)
    frame_ids: list[list[int]] = []
    sum_ray_chk = torch.zeros(N, device=device, dtype=torch.float32)

    print(f"[stage1] back-projecting {len(cams)} frames, one trace per frame "
          f"(one-hot of that frame's ids + a constant-1.0 coverage plane)")
    for i, cam in enumerate(cams):
        H, W = int(cam.image_height), int(cam.image_width)
        m = masks[i].to(device)
        if m.shape != (H, W):
            m = F.interpolate(m[None, None].float(), size=(H, W),
                              mode="nearest")[0, 0].long()
        ids = sorted({int(x) for x in torch.unique(m).tolist() if int(x) != 0})
        frame_ids.append(ids)
        K = len(ids)
        oh = torch.zeros(H, W, K + 1, device=device, dtype=torch.float32)
        for col, gid in enumerate(ids):
            oh[:, :, col] = (m == gid).float()
        oh[:, :, K] = 1.0
        img_mask = torch.ones(H, W, dtype=torch.int32, device=device)
        g, nr, _, _ = trace_single_view_chunked(
            means, quats, scales, opacities, colors,
            oh.contiguous(), img_mask, cam, bg, backend,
        )
        sum_ray_chk += nr.float()
        cover = g[:, K].clamp(min=1e-12)
        share = g[:, :K] / cover.unsqueeze(-1)
        best, lab = share.max(dim=1)
        vis = nr >= args.min_ray
        member = vis & (best > args.share_thresh)
        lab = torch.where(member, lab.to(torch.int16), torch.full_like(lab, -1, dtype=torch.int16))
        frame_labels[i] = lab.cpu()
        frame_vis[i] = vis.cpu()
        print(f"  frame {names[i]:>6}  ids={ids}  visible={int(vis.sum()):,}  "
              f"assigned={int(member.sum()):,}  "
              + " ".join(f"{gid}:{int((lab == c).sum()):,}" for c, gid in enumerate(ids)))

    same_rays = torch.equal(sum_ray_chk.cpu(), num_ray)
    print(f"[assert] per-frame mask traces sum to the .pt's num_ray bit-exactly: "
          f"{same_rays}  (integer ray counts; the only legal == in this script)")
    if not same_rays:
        raise SystemExit("The per-frame mask traces did NOT see the same rays as the "
                         "feature trace -- 'same Gaussians, same cameras' is false.")
    return dict(names=names, frame_ids=frame_ids, frame_labels=frame_labels,
                frame_vis=frame_vis, num_ray=num_ray, valid=valid, N=N)


def pair_overlaps(bp, device="cuda"):
    """Contingency counts for every cross-frame ``(frame,id)`` pair.

    Returns node list and three arrays over node pairs: intersection, the two
    sides' sizes restricted to the *co-visible* Gaussians, and the two sides'
    global sizes.
    """
    labels = bp["frame_labels"]
    vis = bp["frame_vis"]
    frame_ids = bp["frame_ids"]
    nF = labels.shape[0]
    nodes = [(f, gid) for f in range(nF) for gid in frame_ids[f]]
    idx = {(f, c): n for n, (f, c) in enumerate(
        [(f, ci) for f in range(nF) for ci in range(len(frame_ids[f]))])}
    n_nodes = len(nodes)
    inter = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    a_vis = np.zeros((n_nodes, n_nodes), dtype=np.int64)   # |A cap covisible|
    b_vis = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    size = np.array([int((labels[f] == ci).sum())
                     for f in range(nF) for ci in range(len(frame_ids[f]))],
                    dtype=np.int64)

    lab_gpu = [labels[f].to(device).long() for f in range(nF)]
    vis_gpu = [vis[f].to(device) for f in range(nF)]
    for f in range(nF):
        Kf = len(frame_ids[f])
        for g in range(f + 1, nF):
            Kg = len(frame_ids[g])
            co = vis_gpu[f] & vis_gpu[g]
            a = lab_gpu[f]
            b = lab_gpu[g]
            both = co & (a >= 0) & (b >= 0)
            if int(both.sum()) > 0:
                flat = (a[both] * Kg + b[both])
                cnt = torch.bincount(flat, minlength=Kf * Kg).reshape(Kf, Kg).cpu().numpy()
            else:
                cnt = np.zeros((Kf, Kg), dtype=np.int64)
            av = torch.bincount(a[co & (a >= 0)], minlength=Kf).cpu().numpy()
            bv = torch.bincount(b[co & (b >= 0)], minlength=Kg).cpu().numpy()
            for ci in range(Kf):
                ni = idx[(f, ci)]
                for cj in range(Kg):
                    nj = idx[(g, cj)]
                    inter[ni, nj] = inter[nj, ni] = cnt[ci, cj]
                    a_vis[ni, nj] = a_vis[nj, ni] = av[ci]
                    b_vis[ni, nj] = b_vis[nj, ni] = bv[cj]
    return nodes, inter, a_vis, b_vis, size


def components(adj_bool):
    """Connected components of a boolean adjacency matrix -> label per node."""
    n = adj_bool.shape[0]
    comp = np.full(n, -1, dtype=np.int64)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj_bool[u])[0]:
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    return comp, c


def link_report(bp, nodes, inter, a_vis, b_vis, size, thresholds, measure="vis"):
    """Sweep the overlap threshold; report component count and contradictions."""
    n = len(nodes)
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "vis":
            union = a_vis + b_vis - inter
        else:
            union = size[:, None] + size[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    np.fill_diagonal(iou, 0.0)
    frames = np.array([f for f, _ in nodes])

    rows = []
    for t in thresholds:
        adj = iou > t
        comp, nc = components(adj)
        # A component that holds two ids from one frame is a *contradiction*:
        # within a frame SAM asserts those pixels are different objects.
        contra = 0
        contra_detail = []
        sizes3d = []
        for c in range(nc):
            mem = np.nonzero(comp == c)[0]
            fs = frames[mem]
            dup = len(fs) - len(set(fs.tolist()))
            if dup > 0:
                contra += 1
                contra_detail.append((c, len(mem), len(set(fs.tolist()))))
            sizes3d.append(len(mem))
        multi = int(sum(1 for c in range(nc)
                        if len(set(frames[comp == c].tolist())) >= 2))
        rows.append(dict(threshold=float(t), n_components=int(nc),
                         n_multiframe=multi, n_contradictory=int(contra),
                         largest_component_nodes=int(max(sizes3d)),
                         contra_detail=contra_detail))
    return iou, rows


def print_link_table(rows, measure):
    print(f"\n[link] connected components over cross-frame overlap "
          f"(measure = IoU restricted to co-visible Gaussians)" if measure == "vis"
          else "\n[link] connected components over cross-frame overlap "
               "(measure = plain IoU of Gaussian sets)")
    hdr = (f"{'tau':>6} {'components':>11} {'>=2 frames':>11} "
           f"{'contradictory':>14} {'largest(nodes)':>15}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['threshold']:>6.2f} {r['n_components']:>11} {r['n_multiframe']:>11} "
              f"{r['n_contradictory']:>14} {r['largest_component_nodes']:>15}")


# ----------------------------------------------------------- stage 2: physics
def build_decoders(ckpt_path, feat_dim=32, hidden=64, device="cuda"):
    """The three per-property MLPs, lifted out of the Lightning checkpoint.

    Only the decoders are needed: pooling happens strictly before them
    (physgm_dense_readout.py:68-90), so "same MLP, different membership set" is
    literally these three modules applied to two different pooled vectors.
    """
    from src.model.heads.physics.physgm_readout import _make_property_decoder
    from src.dataset.physics.types import PROPERTY_NAMES

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = raw.get("global_step") if isinstance(raw, dict) else None
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    pre = "model.encoder.physics_scheme.physgm_dense_readout.decoders."
    decs = []
    for i in range(len(PROPERTY_NAMES)):
        d = _make_property_decoder(feat_dim, hidden)
        sub = {k[len(pre) + 2:]: v for k, v in sd.items() if k.startswith(pre + str(i) + ".")}
        missing, unexpected = d.load_state_dict(sub, strict=True), None
        decs.append(d.eval().to(device))
    n = sum(1 for k in sd if k.startswith(pre))
    print(f"[head] {len(decs)} property decoders (LayerNorm->Linear(32,64)->GELU->"
          f"Linear(64,2)) from {ckpt_path} (global_step={step}, {n} tensors matched)")
    return decs, step


@torch.no_grad()
def decode(decs, pooled):
    """pooled [K,32] -> (mu [K,P], var [K,P]) in model space."""
    outs = [d(pooled.float()) for d in decs]
    out = torch.stack(outs, dim=1)                       # [K,P,2]
    return out[..., 0], F.softplus(out[..., 1]) + 1e-2


@torch.no_grad()
def pooled_and_stats(feat, num_ray, sel, decs, chunk=200_000, device="cuda"):
    """One instance -> (pooled 32-d, mu, var, mu_spread) with the ticket-01 protocol.

    ``pooled`` is *sum* of ``gau_sem`` over the instance's Gaussians, i.e.
    ``sum num_ray * feat``; the denominator is omitted on purpose -- the
    decoder's first layer is a LayerNorm, invariant to positive scaling, so any
    positive normaliser feeds the MLP the identical vector (ticket 01's
    correction block).  ``mu_spread`` decodes every Gaussian on its own and
    takes the std of ``mu``; it only means anything because of that same
    scale-invariance (the per-Gaussian vectors are ~50x shorter than the
    training-time ones and spread over ~100x in scale).
    """
    idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
    f = feat[idx].to(device)
    w = num_ray[idx].to(device).unsqueeze(-1)
    pooled = (f * w).sum(0, keepdim=True)
    mu, var = decode(decs, pooled)
    mus = []
    for s0 in range(0, f.shape[0], chunk):
        m, _ = decode(decs, f[s0:s0 + chunk])
        mus.append(m)
    mus = torch.cat(mus, 0)
    return (pooled[0].cpu(), mu[0].cpu(), var[0].cpu(), mus.std(0, unbiased=False).cpu(),
            int(idx.numel()))


def si(mu):
    from src.dataset.physics.parsers import physgm_denormalize
    return physgm_denormalize(mu.unsqueeze(0))[0]     # (density, E, nu)


def side_a_partition(bp, comp, node_col, keep, device="cuda"):
    """Vote-based 3D GT partition.

    A Gaussian joins the linked component that claims it in a strict majority of
    the component's own frames *in which that Gaussian is visible at all* -- the
    across-frame analogue of the per-frame ``share > 0.5`` rule.  Normalising by
    co-visibility rather than by the raw frame count matters: a Gaussian seen in
    8 of the 36 frames must not be disqualified from a 36-frame instance just
    because it is invisible in the other 28.  Ties go to the larger fraction.
    """
    labels = bp["frame_labels"]
    vis = bp["frame_vis"]
    N = bp["N"]
    frac = torch.zeros(len(keep), N, dtype=torch.float32)
    for k, c in enumerate(keep):
        votes = torch.zeros(N, dtype=torch.float32)
        denom = torch.zeros(N, dtype=torch.float32)
        for m, (f, ci) in enumerate(node_col):
            if comp[m] != c:
                continue
            votes += (labels[f] == ci).float()
            denom += vis[f].float()
        frac[k] = votes / denom.clamp(min=1)
    best, arg = frac.max(0)
    member = best > 0.5
    return arg, member


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traced", required=True)
    ap.add_argument("--traced_control", default=None)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--hdbscan_labels", default=None)
    ap.add_argument("--phys_ckpt", default=None)
    ap.add_argument("--images_folder", default="images")
    ap.add_argument("--share_thresh", type=float, default=0.5)
    ap.add_argument("--min_ray", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.5, help="linking threshold used for side A")
    ap.add_argument("--min_frames", type=int, default=None,
                    help="a 3D GT instance must be linked across at least this many "
                         "frames (default: all of them -- the strongest statement the "
                         "data can make, and the only setting that is threshold-stable)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        print(f"[stage1] reusing cached back-projection {cache}")
        bp = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        bp = backproject_per_frame(args, device)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bp, cache)

    nodes, inter, a_vis, b_vis, size = pair_overlaps(bp, device)
    frame_ids = bp["frame_ids"]; nF = len(frame_ids)
    node_col = [(f, ci) for f in range(nF) for ci in range(len(frame_ids[f]))]
    print(f"\n[link] {len(nodes)} (frame, id) nodes over {nF} frames; Gaussian-set "
          f"sizes: min={size.min():,} median={int(np.median(size)):,} max={size.max():,}")

    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iou_vis, rows_vis = link_report(bp, nodes, inter, a_vis, b_vis, size, taus, "vis")
    print_link_table(rows_vis, "vis")
    iou_glob, rows_glob = link_report(bp, nodes, inter, a_vis, b_vis, size, taus, "glob")
    print_link_table(rows_glob, "glob")
    np.save(out / "iou_covisible.npy", iou_vis)
    np.save(out / "iou_global.npy", iou_glob)

    # --- how stable is the *core* of the linking?  A component that holds one id
    # from every single frame is the strongest possible statement the data can make.
    frames_arr = np.array([f for f, _ in nodes])
    stab = {}
    for t in taus:
        comp, nc = components(iou_vis > t)
        full = [c for c in range(nc)
                if len(set(frames_arr[comp == c].tolist())) == nF
                and (comp == c).sum() == nF]
        stab[t] = sorted(tuple(sorted(np.nonzero(comp == c)[0].tolist())) for c in full)
    print(f"\n[link] components spanning ALL {nF} frames with exactly one id per frame:")
    ref = stab[0.5]
    for t in taus:
        same = ("  == the tau=0.50 set" if stab[t] == ref
                else "  differs from the tau=0.50 set")
        print(f"  tau={t:<5} {len(stab[t]):>2} such components{same}")
    plateau = [t for t in taus if stab[t] == ref]
    print(f"  -> the same {len(ref)} instances for tau in "
          f"[{min(plateau)}, {max(plateau)}]; outside that window the linking changes")

    min_frames = args.min_frames or nF
    comp, nc = components(iou_vis > args.tau)
    keep = [c for c in range(nc)
            if len(set(frames_arr[comp == c].tolist())) >= min_frames]
    dropped = [c for c in range(nc) if c not in keep]
    n_drop_nodes = int(sum((comp == c).sum() for c in dropped))
    print(f"\n[side A] tau={args.tau}, keeping {len(keep)} of {nc} components "
          f"(linked across >= {min_frames} of {nF} frames); {len(dropped)} components "
          f"/ {n_drop_nodes} (frame,id) nodes dropped as not frame-consistent")
    argc, memberA = side_a_partition(bp, comp, node_col, keep, device)
    print(f"[side A] {int(memberA.sum()):,} Gaussians partitioned into {len(keep)} "
          f"3D GT instances (strict majority of the frames that claim them)")

    ck = torch.load(args.traced, map_location="cpu", weights_only=False)
    feat = ck["gau_phys_feat"]; num_ray = ck["num_ray"]; valid = ck["valid_mask"]
    labB = torch.from_numpy(np.load(args.hdbscan_labels)["labels"].astype(np.int64))
    decs, step = build_decoders(args.phys_ckpt, device=device)

    ctrl = None
    if args.traced_control:
        cc = torch.load(args.traced_control, map_location="cpu", weights_only=False)
        assert torch.equal(cc["num_ray"], num_ray), "control run saw different rays"
        ctrl = cc["gau_phys_feat"]

    def rows_for(sel_of, ids, feat_):
        r = {}
        for k in ids:
            sel = sel_of(k)
            if int(sel.sum()) == 0:
                continue
            p, mu, var, spread, n = pooled_and_stats(feat_, num_ray, sel, decs)
            r[k] = dict(n=n, sel=sel, pooled=p, mu=mu, var=var, spread=spread,
                        si=si(mu))
        return r

    A = rows_for(lambda k: memberA & (argc == keep.index(k)), keep, feat)
    Bids = sorted({int(x) for x in labB.unique().tolist() if x != 0})
    B = rows_for(lambda k: valid & (labB == k), Bids, feat)

    # ------------------------------------------------------------- noise floor
    floor = None
    if ctrl is not None:
        Ac = rows_for(lambda k: memberA & (argc == keep.index(k)), keep, ctrl)
        Bc = rows_for(lambda k: valid & (labB == k), Bids, ctrl)
        d = []
        for src, dst in ((A, Ac), (B, Bc)):
            for k in src:
                a, b = src[k]["si"], dst[k]["si"]
                d.append((2 * (a - b).abs() / (a + b).abs()).tolist())
        floor = np.array(d).max(0)
        print(f"\n[noise floor] same membership set, the SAME trace command run twice: "
              f"max relative difference in (rho, E, nu) = "
              f"{floor[0]:.2e} / {floor[1]:.2e} / {floor[2]:.2e}")

    # ---------------------------------------------------------------- pairing
    print("\n[pairing] rule: each side-A instance takes the side-B instance with the "
          "largest IoU over Gaussian sets (greedy per A row, B may repeat); "
          "A rows with best IoU = 0 are unpaired.")
    pairs = []
    iou_ab = {}
    for ka in A:
        row = {}
        for kb in B:
            inter_ = int((A[ka]["sel"] & B[kb]["sel"]).sum())
            union_ = int((A[ka]["sel"] | B[kb]["sel"]).sum())
            row[kb] = inter_ / union_ if union_ else 0.0
        iou_ab[ka] = row
        top = sorted(row.items(), key=lambda x: -x[1])
        pairs.append((ka, top[0][0] if top[0][1] > 0 else None, top[0][1]))
        print(f"  A{ka}: " + ", ".join(f"B{k}={v:.3f}" for k, v in top[:3]))
    used = {b for _, b, _ in pairs if b is not None}
    print(f"[pairing] {sum(1 for _, b, _ in pairs if b is not None)} of {len(A)} side-A "
          f"instances paired; {len(B) - len(used)} of {len(B)} side-B instances unpaired "
          f"(ids {sorted(set(B) - used)})")

    from src.dataset.physics.types import PROPERTY_NAMES
    print("\n" + "=" * 150)
    print(f"MEMBERSHIP-SET MISMATCH   head = {args.phys_ckpt} (global_step={step})")
    print("  A = 3D GT instance (linked sam/mask ids)   B = HDBSCAN instance "
          "(ticket 03 labels)   rho kg/m3, E Pa, nu")
    print("  var = the head's own softplus variance (model space); mu_spread = std of "
          "per-Gaussian mu (model space)")
    print("=" * 150)
    hdr = (f"{'A':>3} {'B':>3} {'IoU':>6} {'nA':>9} {'nB':>9} | "
           f"{'rhoA':>9} {'rhoB':>9} {'d_rho':>7} | {'E_A':>10} {'E_B':>10} {'d_E':>7} | "
           f"{'nuA':>6} {'nuB':>6} {'d_nu':>7} | {'varA':>17} {'varB':>17} | "
           f"{'spreadA':>17} {'spreadB':>17}")
    print(hdr); print("-" * len(hdr))
    table = []
    for ka, kb, iou_ in sorted(pairs, key=lambda x: -A[x[0]]["n"]):
        a = A[ka]
        if kb is None:
            print(f"{ka:>3} {'-':>3} {0.0:>6.3f} {a['n']:>9,} {'-':>9}  (unpaired)")
            continue
        b = B[kb]
        rel = (2 * (a["si"] - b["si"]).abs() / (a["si"] + b["si"]).abs())
        print(f"{ka:>3} {kb:>3} {iou_:>6.3f} {a['n']:>9,} {b['n']:>9,} | "
              f"{a['si'][0]:>9.1f} {b['si'][0]:>9.1f} {rel[0]:>7.4f} | "
              f"{a['si'][1]:>10.3e} {b['si'][1]:>10.3e} {rel[1]:>7.4f} | "
              f"{a['si'][2]:>6.4f} {b['si'][2]:>6.4f} {rel[2]:>7.4f} | "
              + " ".join(f"{v:.3f}" for v in a["var"].tolist()) + "  "
              + " ".join(f"{v:.3f}" for v in b["var"].tolist()) + "  | "
              + " ".join(f"{v:.3f}" for v in a["spread"].tolist()) + "  "
              + " ".join(f"{v:.3f}" for v in b["spread"].tolist()))
        table.append(dict(a=int(ka), b=int(kb), iou=iou_, n_a=a["n"], n_b=b["n"],
                          si_a=a["si"].tolist(), si_b=b["si"].tolist(),
                          rel=rel.tolist(), var_a=a["var"].tolist(),
                          var_b=b["var"].tolist(), spread_a=a["spread"].tolist(),
                          spread_b=b["spread"].tolist()))
    print("=" * 150)
    print(f"property order: {PROPERTY_NAMES}")

    # The SI relative difference is not scale-free: density and E are decoded
    # through 10**(z*std + mean) with std 0.399 / 1.318, so a small shift of the
    # pooled vector's mu is amplified by a factor of ~e^(2.3*std) in SI.  The
    # model-space |mu_A - mu_B| (in z units) is what the head actually moved, and
    # dividing it by that instance's mu_spread says whether the two membership
    # sets disagree by more than the instance disagrees with itself.
    print("\nSAME ROWS IN MODEL SPACE (z units): |mu_A - mu_B|, and that over side A's "
          "mu_spread")
    h2 = (f"{'A':>3} {'B':>3} {'nA':>9} | " + " ".join(f"{'dmu_' + n[:5]:>10}" for n in PROPERTY_NAMES)
          + " | " + " ".join(f"{'/spread_' + n[:5]:>13}" for n in PROPERTY_NAMES))
    print(h2); print("-" * len(h2))
    for r in table:
        dmu = (torch.tensor(r["si_a"]), torch.tensor(r["si_b"]))
        a = A[r["a"]]; b = B[r["b"]]
        d = (a["mu"] - b["mu"]).abs()
        ratio = d / a["spread"].clamp(min=1e-9)
        r["dmu_model"] = d.tolist(); r["dmu_over_spread"] = ratio.tolist()
        print(f"{r['a']:>3} {r['b']:>3} {r['n_a']:>9,} | "
              + " ".join(f"{v:>10.4f}" for v in d.tolist()) + " | "
              + " ".join(f"{v:>13.4f}" for v in ratio.tolist()))

    # ---------------------------------------------------- what actually differs
    print("\n[composition] each side-A instance broken down by the HDBSCAN label its "
          "Gaussians carry (fractions >= 0.5%)")
    for ka in A:
        sel = A[ka]["sel"]
        frac = {int(b): round(float((sel & B[b]["sel"]).sum()) / float(sel.sum()), 4)
                for b in B if float((sel & B[b]["sel"]).sum()) > 0.005 * float(sel.sum())}
        print(f"  A{ka} n={A[ka]['n']:,}  {frac}")

    print("\n[disagreement] how big is the symmetric difference, and how much of each "
          "side's pooling weight (sum num_ray) does it carry?")
    h3 = (f"{'A':>3} {'B':>3} {'|A xor B|':>10} {'% of union':>11} "
          f"{'% weight of A':>14} {'% weight of B':>14}")
    print(h3); print("-" * len(h3))
    for ka, kb, _ in sorted(pairs, key=lambda x: -A[x[0]]["n"]):
        if kb is None:
            continue
        sA, sB = A[ka]["sel"], B[kb]["sel"]
        x = sA ^ sB
        wa = float(num_ray[sA].sum()); wb = float(num_ray[sB].sum())
        print(f"{ka:>3} {kb:>3} {int(x.sum()):>10,} "
              f"{100 * int(x.sum()) / int((sA | sB).sum()):>10.1f}% "
              f"{100 * float(num_ray[sA & x].sum()) / wa:>13.1f}% "
              f"{100 * float(num_ray[sB & x].sum()) / wb:>13.1f}%")

    print("\n[pairing sensitivity] ALTERNATIVE RULE -- a side-A instance takes the "
          "UNION of every HDBSCAN instance that is >= 20% inside it (split-aware, "
          "many-to-one).  Only rows where HDBSCAN split a GT object can change.")
    h4 = (f"{'A':>3} {'B set':>14} {'nB':>9} | {'d_rho':>7} {'d_E':>7} {'d_nu':>7}")
    print(h4); print("-" * len(h4))
    alt = []
    for ka in sorted(A, key=lambda k: -A[k]["n"]):
        sel = A[ka]["sel"]
        bs = [b for b in B if float((sel & B[b]["sel"]).sum()) > 0.20 * float(B[b]["sel"].sum())]
        u = torch.zeros_like(sel)
        for b in bs:
            u |= B[b]["sel"]
        u &= valid
        _, mu, var, spread, n = pooled_and_stats(feat, num_ray, u, decs)
        sb = si(mu)
        rel = (2 * (A[ka]["si"] - sb).abs() / (A[ka]["si"] + sb).abs())
        print(f"{ka:>3} {str(bs):>14} {n:>9,} | " + " ".join(f"{v:>7.4f}" for v in rel.tolist()))
        alt.append(dict(a=int(ka), b_set=[int(x) for x in bs], n_b=n,
                        si_b=sb.tolist(), rel=rel.tolist()))

    (out / "membership_table.json").write_text(json.dumps(dict(
        head=args.phys_ckpt, global_step=step, tau=args.tau,
        share_thresh=args.share_thresh, min_ray=args.min_ray,
        noise_floor_si_rel=None if floor is None else floor.tolist(),
        sweep_covisible_iou=rows_vis, sweep_global_iou=rows_glob,
        full_span_components={str(t): len(v) for t, v in stab.items()},
        rows=table,
        alt_pairing=alt,
        side_b_all={int(k): dict(n=B[k]["n"], si=B[k]["si"].tolist(),
                                 var=B[k]["var"].tolist(),
                                 spread=B[k]["spread"].tolist()) for k in B},
    ), indent=2, default=str))
    print(f"\n[out] {out / 'membership_table.json'}")


if __name__ == "__main__":
    main()

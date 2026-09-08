#!/usr/bin/env python3
"""Acceptance harness for ticket 04 of the iggt-phys-pipeline map.

Answers two questions about the `.pt` written by
``trace_instance_to_gaussians.py --feature_source iggt_phys``:

1. **Are the two streams really on the same Gaussians and the same cameras?**
   The trace loop already asserts it per view (bit-exact ``num_ray``); here it is
   re-checked from disk, against a *control group* (the same path run twice), as
   the map's fact F4 demands -- ``atomicAdd`` makes the kernel non-deterministic,
   so nothing may be compared with ``torch.equal`` except integer ray counts.

2. **How far apart are the 2D and the 3D pooling protocols?**  (ticket 01's
   "inequivalence 1", assigned here because this is the only place that holds
   both the 2D ``feat_map`` and the 3D ``gau_phys_feat``.)  Under *one and the
   same* instance mask:
     - 2D protocol  = ``pool_one_sample`` over the dense map, uniform per pixel
       across views -- this is literally the training protocol
       (``physics_pool.py:79-80``);
     - 3D protocol  = pooled over that instance's Gaussians, both ``num_ray``-
       weighted (ticket 01's choice: ``sum num_ray*feat / sum num_ray`` is the
       faithful analogue of the pixel mean) and equal-weight (the control).

   The instance mask is the scene's GT ``sam/mask/*.png``.  Its 3D counterpart
   is obtained by tracing the GT mask's one-hot encoding through the *same*
   cameras and the *same* Gaussians, so "same mask" holds on both sides by
   construction rather than by resemblance.

Usage (bench):

    CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 python scripts/check_iggt_phys_trace.py \
        --traced <out>/gaussian_iggt_phys_feat.pt \
        --traced_control <out2>/gaussian_iggt_phys_feat.pt \
        --mask_dir <scene>/sam/mask \
        --iggt_model_path /home/liaoyuanjun/iggt_checkpoint.pth \
        --iggt_phys_ckpt <physhead>.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.trace_instance_to_gaussians import prepare_iggt_phys_features  # noqa: E402
from src.model.heads.physics.physics_pool import pool_one_sample  # noqa: E402
from src.trace_cameras import load_trace_cameras  # noqa: E402
from src.trace_render.trace_rasterize import (  # noqa: E402
    load_gaussians_from_ply,
    resolve_trace_backend,
    trace_single_view_chunked,
)


def _layernorm(x: torch.Tensor) -> torch.Tensor:
    """What the PhysGM decoder's first layer does to its input (LayerNorm, no affine)."""
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-5)


def cos_and_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """``(cosine, post-LayerNorm cosine, relative L2)``.

    Relative L2 is ``||a-b|| / (0.5*(||a||+||b||))``; it saturates at 2.0 when the
    two vectors differ mostly in magnitude, so it is reported *next to* the two
    cosines rather than instead of them.  The post-LayerNorm cosine is the one
    that matters downstream: the per-property decoder starts with a LayerNorm
    (physgm_readout.py:27-37), which removes exactly the per-vector offset and
    scale, so only this quantity survives into the MLP.
    """
    a = a.double()
    b = b.double()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))
    an, bn = _layernorm(a), _layernorm(b)
    cos_ln = float(torch.dot(an, bn) / (an.norm() * bn.norm() + 1e-30))
    rel = float((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-30))
    return cos, cos_ln, rel


def load_gt_masks(mask_dir, image_names):
    """Read one integer instance mask per view, in camera order → list of [H,W] long."""
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
        if hit.suffix == ".npy":
            arr = np.load(hit)
        else:
            arr = np.array(Image.open(hit))
        if arr.ndim != 2:
            raise ValueError(f"{hit}: expected a single-channel id map, got {arr.shape}")
        out.append(torch.from_numpy(arr.astype(np.int64)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traced", required=True, help="the iggt_phys .pt under test")
    ap.add_argument("--traced_control", default=None,
                    help="a second .pt from the SAME command (the noise floor)")
    ap.add_argument("--mask_dir", required=True, help="GT instance masks, one per view")
    ap.add_argument("--iggt_model_path", required=True)
    ap.add_argument("--iggt_phys_ckpt", default=None)
    ap.add_argument("--dumped_feat_maps", default=None,
                    help="directory written by --dump_feat_maps (adds an fp16-vs-fp32 "
                         "floor for the 2D side)")
    ap.add_argument("--images_folder", default="images")
    ap.add_argument("--encoder_batch_size", type=int, default=4)
    ap.add_argument("--max_views", type=int, default=None)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.traced, map_location="cpu", weights_only=False)
    prov = ck.get("provenance", {})
    print("=" * 78)
    print(f"traced .pt : {args.traced}")
    print(f"  feature_source={ck['feature_source']}  n_views={ck['n_views']}  "
          f"trace_backend={ck['trace_backend']}")
    for k, v in prov.items():
        print(f"  {k}: {v}")

    inst3d = ck["gau_inst_feat"]
    phys3d = ck["gau_phys_feat"]
    num_ray = ck["num_ray"]
    valid = ck["valid_mask"]
    N = inst3d.shape[0]
    print(f"\n[coverage] {int(valid.sum()):,} / {N:,} Gaussians traced "
          f"({100 * float(valid.float().mean()):.1f}%)")
    print(f"[shapes]   gau_inst_feat {tuple(inst3d.shape)}  "
          f"gau_phys_feat {tuple(phys3d.shape)}  num_ray {tuple(num_ray.shape)}  "
          f"gaussian_index {tuple(ck['gaussian_index'].shape)}")

    # ---------------------------------------------------------------- 1. control
    floor_inst = floor_phys = None
    if args.traced_control:
        cc = torch.load(args.traced_control, map_location="cpu", weights_only=False)
        print(f"\n[control] second run of the same path: {args.traced_control}")
        for name, a, b in (("gau_inst_feat", inst3d, cc["gau_inst_feat"]),
                           ("gau_phys_feat", phys3d, cc["gau_phys_feat"])):
            d = (a - b).abs()
            scale = a.abs().max().item()
            rel = d.max().item() / max(scale, 1e-30)
            print(f"  {name:14s} maxabs={d.max().item():.3e}  mean={d.mean().item():.3e}"
                  f"  |x|max={scale:.3e}  rel={rel:.3e}")
            if name == "gau_inst_feat":
                floor_inst = cc["gau_inst_feat"]
            else:
                floor_phys = cc["gau_phys_feat"]
        print(f"  num_ray bit-exact across runs: {torch.equal(num_ray, cc['num_ray'])} "
              "(integer ray counts are deterministic; features are not)")

    # ------------------------------------------------- 2. geometry / GT mask → 3D
    means, quats, scales, opacities, colors = load_gaussians_from_ply(ck["ply_path"])
    backend = resolve_trace_backend("auto", int(scales.shape[1]))
    cams = load_trace_cameras(ck.get("camera_backend", "colmap"),
                              source_path=ck["source_path"],
                              images_folder=args.images_folder,
                              resolution=1,
                              transforms_json=ck.get("transforms_json"),
                              transforms_axis=ck.get("transforms_axis", "nerfstudio"))
    n_views = args.max_views or ck["n_views"]
    cams = cams[:n_views]
    names = [c.image_name for c in cams]

    masks = load_gt_masks(args.mask_dir, names)
    ids = sorted({int(i) for m in masks for i in torch.unique(m).tolist()})
    fg = [i for i in ids if i != 0]
    print(f"\n[mask] {args.mask_dir}: ids={ids} (0 treated as background/ignore) "
          f"→ {len(fg)} instances, native {tuple(masks[0].shape)}")

    K = len(fg)
    id_to_col = {i: c for c, i in enumerate(fg)}
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Channel K is a constant 1.0 "coverage" plane.  It is NOT decoration: the trace
    # kernel accumulates gau_sem[g] = sum_r w_gr * img_sem[r] with per-ray blend
    # weights w (alpha * transmittance), while num_ray[g] is an unweighted hit
    # count -- so gau_sem/num_ray is a weighted sum over an unweighted count, not a
    # per-Gaussian pixel mean (measured: a constant-1.0 map traces to a median of
    # 0.011, not 1.0).  Dividing the one-hot channels by this coverage channel is
    # the only way to read a real per-Gaussian instance *share* in [0, 1].
    n_ch = K + 1
    sum_onehot = torch.zeros(means.shape[0], n_ch, device=device, dtype=torch.float32)
    sum_ray_chk = torch.zeros(means.shape[0], device=device, dtype=torch.float32)
    print(f"[mask→3D] tracing the GT mask one-hot + coverage ({n_ch} channels) "
          f"through the same {len(cams)} cameras and the same Gaussians ...")
    for i, cam in enumerate(cams):
        H, W = int(cam.image_height), int(cam.image_width)
        m = masks[i].to(device)
        if m.shape != (H, W):
            m = F.interpolate(m[None, None].float(), size=(H, W),
                              mode="nearest")[0, 0].long()
        oh = torch.zeros(H, W, n_ch, device=device, dtype=torch.float32)
        for gid, col in id_to_col.items():
            oh[:, :, col] = (m == gid).float()
        oh[:, :, K] = 1.0
        img_mask = torch.ones(H, W, dtype=torch.int32, device=device)
        g, nr, _, _ = trace_single_view_chunked(
            means, quats, scales, opacities, colors,
            oh.contiguous(), img_mask, cam, bg, backend,
        )
        sum_onehot += g
        sum_ray_chk += nr.float()

    same_rays = torch.equal(sum_ray_chk.cpu(), num_ray)
    print(f"[assert] mask trace saw the same rays as the feature trace: {same_rays}")
    if not same_rays:
        raise SystemExit(
            "num_ray from the GT-mask trace differs from the .pt: the mask was NOT "
            "reprojected through the same cameras/Gaussians, so 'same mask' is false."
        )

    occ = sum_onehot.cpu()
    cover = occ[:, K].clamp(min=1e-12)          # sum_r w_gr, the Gaussian's blend mass
    scale = torch.zeros(N)
    scale[valid] = occ[valid, K] / num_ray[valid]
    print(f"[kernel] blend-mass / ray-count  (a constant-1.0 map traced and divided "
          f"by num_ray): p05={scale[valid].quantile(0.05):.4f} "
          f"p50={scale[valid].quantile(0.5):.4f} "
          f"p95={scale[valid].quantile(0.95):.4f} max={scale[valid].max():.4f}")
    if "blend_mass" in ck:
        # Independent redo of the .pt's own blend-mass pass: same quantity, traced
        # again here from a different call site.  Control-grouped, never ==.
        bm = ck["blend_mass"]
        d = (bm - occ[:, K]).abs()
        print(f"  vs the .pt's saved blend_mass: maxabs={d.max():.3e}  "
              f"mean={d.mean():.3e}  |x|max={bm.abs().max():.3e}  "
              f"rel={d.max() / bm.abs().max():.3e}")
    share = occ[:, :K] / cover.unsqueeze(-1)
    best, label = share.max(dim=1)
    # A Gaussian belongs to instance k when k wins the reprojected mask vote and owns
    # more than half of the Gaussian's blend mass.  Background/stuff Gaussians (the
    # GT mask leaves 0 there) drop out here rather than joining a random instance.
    member = valid & (best > 0.5)
    print(f"[mask→3D] {int(member.sum()):,} of {int(valid.sum()):,} traced Gaussians "
          f"assigned to an instance (>50% of their blend mass)")

    # -------------------------------------------------------- 3. 2D protocol (fp32)
    class _A:
        pass
    a2 = _A()
    a2.iggt_model_path = args.iggt_model_path
    a2.iggt_phys_ckpt = args.iggt_phys_ckpt
    a2._iggt_phys_image_size = tuple(prov.get("image_size_wh", (504, 336)))
    a2.encoder_batch_size = prov.get("encoder_batch_size", args.encoder_batch_size)
    a2.trace_crop_aligned = False
    print("\n[2D] re-running the IGGT forward in fp32 for the 2D pooling protocol "
          "(same weights, same preprocessing as the traced run) ...")
    prep = prepare_iggt_phys_features(cams, a2, device)
    fm = torch.stack(prep["aux_feat_maps"]["phys"], dim=0).to(device)  # [V,32,h,w]
    _, _, fh, fw = fm.shape
    mask_small = torch.stack([
        F.interpolate(m[None, None].float(), size=(fh, fw), mode="nearest")[0, 0].long()
        for m in masks
    ], dim=0).to(device)
    pooled2d, ids2d = pool_one_sample(fm, mask_small, None, ignore_id=0)
    pooled2d = pooled2d.cpu()
    ids2d = ids2d.cpu().tolist()
    print(f"[2D] pool_one_sample over [{fm.shape[0]},32,{fh},{fw}] → "
          f"{tuple(pooled2d.shape)} for ids {ids2d}")

    floor2d = None
    if args.dumped_feat_maps:
        d = Path(args.dumped_feat_maps) / "phys"
        fm16 = torch.stack([torch.load(d / f"{n}.pt", map_location=device).float()
                            for n in names], dim=0)
        p16, _ = pool_one_sample(fm16, mask_small, None, ignore_id=0)
        floor2d = p16.cpu()
        del fm16

    # --------------------------------------------------------------- 4. comparison
    def pool3d(sel, weighted):
        f = phys3d[sel]
        if not weighted:
            return f.mean(dim=0)
        w = num_ray[sel].unsqueeze(-1)
        return (f * w).sum(dim=0) / w.sum()

    print("\n" + "=" * 78)
    print("POOLING-PROTOCOL MISMATCH  (mask = GT sam/mask, same mask on both sides)")
    print("  2D  = pool_one_sample over the dense 32-d map, uniform per pixel across")
    print("        views  = the training protocol (physics_pool.py:79-80)")
    print("  3Dw = sum(num_ray * feat) / sum(num_ray) over that instance's Gaussians")
    print("  3De = plain mean over that instance's Gaussians")
    print("  cos = raw cosine;  cosLN = cosine after LayerNorm (what the decoder")
    print("        actually sees);  rel = relative L2, saturates at 2.0 when the")
    print("        two vectors differ mainly in magnitude")
    print("=" * 78)
    cols = ("cos", "cosLN", "rel")
    hdr = (f"{'inst':>5} {'n_gau':>9} {'n_pix2D':>9} | "
           + " ".join(f"{c + ' 2D-3Dw':>12}" for c in cols) + " | "
           + " ".join(f"{c + ' 2D-3De':>12}" for c in cols) + " | "
           + " ".join(f"{c + ' w-e':>10}" for c in cols)
           + f" | {'|3Dw|/|2D|':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for gid in fg:
        if gid not in ids2d:
            print(f"{gid:>5} (absent from the 2D pool at feature resolution)")
            continue
        v2 = pooled2d[ids2d.index(gid)]
        sel = member & (label == id_to_col[gid])
        n_g = int(sel.sum())
        n_p = int((mask_small == gid).sum())
        if n_g == 0:
            print(f"{gid:>5} {n_g:>9} {n_p:>9}  (no Gaussians won this instance)")
            continue
        vw = pool3d(sel, True)
        ve = pool3d(sel, False)
        m = (*cos_and_rel(v2, vw), *cos_and_rel(v2, ve), *cos_and_rel(vw, ve),
             float(vw.norm() / (v2.norm() + 1e-30)))
        rows.append((gid, n_g, n_p, *m))
        print(f"{gid:>5} {n_g:>9,} {n_p:>9,} | "
              + " ".join(f"{m[i]:>12.4f}" for i in (0, 1, 2)) + " | "
              + " ".join(f"{m[i]:>12.4f}" for i in (3, 4, 5)) + " | "
              + " ".join(f"{m[i]:>10.4f}" for i in (6, 7, 8))
              + f" | {m[9]:>10.4f}")
    if rows:
        a = np.array([r[3:] for r in rows]).mean(0)
        print("-" * len(hdr))
        print(f"{'mean':>5} {'':>9} {'':>9} | "
              + " ".join(f"{a[i]:>12.4f}" for i in (0, 1, 2)) + " | "
              + " ".join(f"{a[i]:>12.4f}" for i in (3, 4, 5)) + " | "
              + " ".join(f"{a[i]:>10.4f}" for i in (6, 7, 8))
              + f" | {a[9]:>10.4f}")

    # ---------------------------------------------------------------- noise floors
    print("\nNOISE FLOORS (nothing above is real unless it clears these)")
    if not rows:
        print("  (no instance produced both a 2D and a 3D pool; nothing to floor)")
    if rows and floor_phys is not None:
        cs, rs = [], []
        for gid, *_ in rows:
            sel = member & (label == id_to_col[gid])
            w = num_ray[sel].unsqueeze(-1)
            v1 = (phys3d[sel] * w).sum(0) / w.sum()
            v2 = (floor_phys[sel] * w).sum(0) / w.sum()
            c, cln, r = cos_and_rel(v1, v2)
            cs.append(min(c, cln))
            rs.append(r)
        print(f"  3D side  (same path twice, num_ray-weighted per instance): "
              f"cos/cosLN min={min(cs):.6f}  relL2 max={max(rs):.3e}")
    if rows and floor2d is not None:
        cs, rs = [], []
        for gid, *_ in rows:
            c, cln, r = cos_and_rel(pooled2d[ids2d.index(gid)],
                                    floor2d[ids2d.index(gid)])
            cs.append(min(c, cln))
            rs.append(r)
        print(f"  2D side  (fp32 re-forward vs the fp16 map that was traced): "
              f"cos/cosLN min={min(cs):.6f}  relL2 max={max(rs):.3e}")
    print("=" * 78)


if __name__ == "__main__":
    main()

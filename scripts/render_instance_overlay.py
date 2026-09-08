"""Full-resolution instance overlay: the original photo with semi-transparent
per-instance colours composited on top.

This is the acceptance figure. It deliberately does **not** produce a grey- or
white-background 3DGS render of the coloured Gaussians: the predecessor effort's
ticket 04 was rejected at review for exactly that (a quarter-resolution render
whose background occupied 87.8% of the frame), while the same labels overlaid
back onto the original images tracked object boundaries correctly.

How the per-pixel instance map is obtained
------------------------------------------
Alpha-blending LUT colours would muddy boundaries, so instead the rasterizer is
run once per group of three instances with ``colors_precomp`` set to a one-hot
vector. With a black background each output channel is then exactly the
alpha-accumulated coverage of one instance:

    out_color[c] = sum_i  alpha_i * T_i * onehot_i[c]

Stacking those gives a per-instance coverage volume ``[K, H, W]``; ``argmax``
over K is a crisp per-pixel instance id, and the sum over K is the total
splat coverage, which drives the compositing alpha (so unmodelled regions such
as sky keep the original photo instead of turning grey).

Usage
-----
    python scripts/render_instance_overlay.py \
        --source_path .../3dovs/bench \
        --ply .../3dovs/bench/point_cloud.ply \
        --labels .../bench_labels_default.npz \
        --out_dir .../overlay --n_views 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trace_cameras import load_trace_cameras
from src.trace_render.trace_rasterize import (
    TRACE_CHANNELS,
    load_gaussians_from_ply,
    resolve_trace_backend,
    trace_single_view,
)
from src.visualization.instance_viz import make_color_lut


def distinct_palette(n: int) -> np.ndarray:
    """``n`` visually separable RGB colours (golden-ratio hue walk, alternating
    S/V). ``make_color_lut`` draws uniformly at random and on bench handed the
    two biggest clusters two near-identical greens, which makes the acceptance
    figure unreadable."""
    import colorsys

    cols = []
    golden = 0.6180339887498949
    hue = 0.03
    for i in range(n):
        s = (0.95, 0.62, 0.80)[i % 3]
        v = (1.00, 0.90, 0.72)[i % 3]
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, s, v)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
        hue += golden
    return np.asarray(cols, dtype=np.uint8)


def load_labels(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        z = np.load(path)
        return z["labels"].astype(np.int32)
    return np.load(path).astype(np.int32)


@torch.no_grad()
def instance_coverage(means, quats, scales, opacities, cam, labels_t, n_labels,
                      backend, device="cuda"):
    """Return ``cov [K, H, W]`` = per-instance alpha coverage, K = n_labels."""
    h, w = int(cam.image_height), int(cam.image_width)
    img_sem = torch.zeros(h, w, TRACE_CHANNELS, device=device, dtype=torch.float32)
    img_mask = torch.ones(h, w, dtype=torch.int32, device=device)
    bg = torch.zeros(3, dtype=torch.float32, device=device)

    cov = torch.zeros(n_labels, h, w, device=device, dtype=torch.float32)
    g = labels_t.shape[0]
    for start in range(0, n_labels, 3):
        stop = min(start + 3, n_labels)
        colors = torch.zeros(g, 3, device=device, dtype=torch.float32)
        for j, lab in enumerate(range(start, stop)):
            colors[:, j] = (labels_t == (lab + 1)).float()  # labels are 1..K
        _a, _b, _r, out_color = trace_single_view(
            means, quats, scales, opacities, colors,
            img_sem, img_mask, cam, bg, backend,
        )
        cov[start:stop] = out_color[: stop - start]
        del colors
    return cov


def boundary_mask(lbl: np.ndarray) -> np.ndarray:
    """1-pixel boundaries of a label image."""
    b = np.zeros_like(lbl, dtype=bool)
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    b[1:, :] |= lbl[:-1, :] != lbl[1:, :]
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    b[:, 1:] |= lbl[:, :-1] != lbl[:, 1:]
    return b


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        d = out.copy()
        d[:-1, :] |= out[1:, :]
        d[1:, :] |= out[:-1, :]
        d[:, :-1] |= out[:, 1:]
        d[:, 1:] |= out[:, :-1]
        out = d
    return out


def _legend_strip(lut: np.ndarray, shares: np.ndarray, width: int,
                  height: int = 46) -> np.ndarray:
    """Colour chips with instance id and mean pixel share."""
    from PIL import ImageDraw

    n = lut.shape[0] - 1
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    cw = width / max(n, 1)
    for i in range(1, n + 1):
        x0 = (i - 1) * cw
        d.rectangle([x0, 0, x0 + cw - 2, height * 0.55],
                    fill=tuple(int(v) for v in lut[i]))
        d.text((x0 + 3, height * 0.60), f"#{i} {shares[i] * 100:.1f}%",
               fill=(0, 0, 0))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--camera_backend", default="colmap")
    ap.add_argument("--images_folder", default="images")
    ap.add_argument("--transforms_json", default=None)
    ap.add_argument("--transforms_axis", default="nerfstudio")
    ap.add_argument("--resolution", type=int, default=1,
                    help="1 = full resolution (what the acceptance figure needs)")
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--view_names", default=None,
                    help="Comma-separated image names; overrides --n_views")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--cov_threshold", type=float, default=0.15,
                    help="Below this total coverage the photo is left untouched")
    ap.add_argument("--palette_seed", type=int, default=0)
    ap.add_argument("--palette", choices=["distinct", "lut"], default="distinct",
                    help="distinct = separable hues (default); lut = make_color_lut")
    ap.add_argument("--outline", type=int, default=2, help="Boundary width in px (0=off)")
    ap.add_argument("--no_sidebyside", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    labels = load_labels(args.labels)
    n_labels = int(labels.max())
    print(f"[labels] {args.labels}: {n_labels} instances "
          f"({int((labels == 0).sum()):,} gaussians unassigned)")

    means, quats, scales, opacities, colors_rgb = load_gaussians_from_ply(args.ply)
    if means.shape[0] != labels.shape[0]:
        raise SystemExit(f"PLY has {means.shape[0]} gaussians, labels has {labels.shape[0]}")
    backend = resolve_trace_backend("auto", int(scales.shape[1]))
    labels_t = torch.from_numpy(labels).to(device)

    cams = load_trace_cameras(
        args.camera_backend, source_path=os.path.abspath(args.source_path),
        images_folder=args.images_folder, resolution=args.resolution,
        transforms_json=args.transforms_json, transforms_axis=args.transforms_axis,
    )
    if args.view_names:
        want = [s.strip() for s in args.view_names.split(",") if s.strip()]
        sel = [c for c in cams if c.image_name in want]
        missing = set(want) - {c.image_name for c in sel}
        if missing:
            raise SystemExit(f"views not found: {sorted(missing)}")
    else:
        n = min(args.n_views, len(cams))
        step = max(1, len(cams) // n)
        sel = [cams[i * step] for i in range(n)]
    print(f"[views] {len(sel)} of {len(cams)}: {[c.image_name for c in sel]}")

    if args.palette == "lut":
        lut = make_color_lut(n_labels + 1, seed=args.palette_seed).astype(np.float32)
    else:
        lut = np.zeros((n_labels + 1, 3), dtype=np.float32)
        lut[1:] = distinct_palette(n_labels)

    per_view = []
    for cam in sel:
        h, w = int(cam.image_height), int(cam.image_width)
        cov = instance_coverage(means, quats, scales, opacities, cam,
                                labels_t, n_labels, backend, device)
        total = cov.sum(0)
        best = cov.argmax(0)
        lbl = (best + 1).to(torch.int32)
        lbl[total < args.cov_threshold] = 0
        lbl_np = lbl.cpu().numpy()
        alpha_np = total.clamp(0, 1).cpu().numpy()

        photo = Image.open(cam.image_path).convert("RGB")
        if photo.size != (w, h):
            photo = photo.resize((w, h), Image.LANCZOS)
        photo_np = np.asarray(photo).astype(np.float32)

        col = lut[np.clip(lbl_np, 0, n_labels)]
        a = (args.alpha * alpha_np * (lbl_np > 0))[..., None]
        out = photo_np * (1.0 - a) + col * a

        if args.outline > 0:
            edge = dilate(boundary_mask(lbl_np) & (lbl_np > 0), args.outline - 1)
            out[edge] = col[edge]

        out_u8 = out.clip(0, 255).astype(np.uint8)
        p_overlay = os.path.join(args.out_dir, f"{cam.image_name}_overlay.png")
        Image.fromarray(out_u8).save(p_overlay)
        if not args.no_sidebyside:
            Image.fromarray(
                np.concatenate([photo_np.astype(np.uint8), out_u8], axis=1)
            ).save(os.path.join(args.out_dir, f"{cam.image_name}_sidebyside.png"))
        np.save(os.path.join(args.out_dir, f"{cam.image_name}_labelmap.npy"),
                lbl_np.astype(np.int16))

        px_ids, px_cnt = np.unique(lbl_np, return_counts=True)
        stat = {
            "view": cam.image_name, "H": h, "W": w,
            "covered_frac": float((lbl_np > 0).mean()),
            "pixel_hist": {int(i): int(c) for i, c in zip(px_ids, px_cnt)},
        }
        per_view.append(stat)
        print(f"  {cam.image_name}: {w}x{h}  covered={stat['covered_frac']*100:.1f}%  "
              f"instances visible={int((px_cnt[px_ids > 0] > 0.001 * h * w).sum())}")
        del cov, total, best

    # ---- legend strip + contact sheet ----
    shares = np.zeros(n_labels + 1)
    for st in per_view:
        tot = st["H"] * st["W"]
        for i, c in st["pixel_hist"].items():
            shares[int(i)] += c / tot / len(per_view)
    legend = _legend_strip(lut, shares, width=sel[0].image_width)
    Image.fromarray(legend).save(os.path.join(args.out_dir, "legend.png"))

    tiles = [np.asarray(Image.open(
        os.path.join(args.out_dir, f"{c.image_name}_overlay.png"))) for c in sel]
    ncol = 2 if len(tiles) <= 4 else 3
    rows = []
    for i in range(0, len(tiles), ncol):
        chunk = tiles[i:i + ncol]
        while len(chunk) < ncol:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    sheet = np.concatenate(rows, axis=0)
    leg = np.asarray(Image.fromarray(legend).resize(
        (sheet.shape[1], int(legend.shape[0] * sheet.shape[1] / legend.shape[1])),
        Image.NEAREST))
    Image.fromarray(np.concatenate([sheet, leg], axis=0)).save(
        os.path.join(args.out_dir, "contact_sheet.png"))

    summary = {
        "labels": os.path.abspath(args.labels),
        "ply": os.path.abspath(args.ply),
        "source_path": os.path.abspath(args.source_path),
        "n_instances": n_labels,
        "resolution": args.resolution,
        "alpha": args.alpha,
        "cov_threshold": args.cov_threshold,
        "palette": {int(i): [int(x) for x in lut[i]] for i in range(n_labels + 1)},
        "views": per_view,
    }
    with open(os.path.join(args.out_dir, "overlay_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[done] -> {args.out_dir}")


if __name__ == "__main__":
    main()

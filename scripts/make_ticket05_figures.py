#!/usr/bin/env python3
"""Assemble the ticket-05 delivery figures from artefacts that already exist.

Three panels, none of which re-runs the model:

``sidebyside``  the official-IGGT instance baseline (ticket 03) next to this
                pipeline's result, same scene, same views, same renderer.  This
                is the panel the map's Destination is judged on ("not obviously
                worse", by eye, no IoU threshold).  A per-view pixel-difference
                count is printed and stamped on the figure so the reader is not
                asked to take "identical" on trust.

``physics``     one full-resolution overlay with a numbered callout per instance
                and the instance physics table rendered beside it: every 3D
                instance visibly carries (E, nu, rho), with n_gau and the
                small-instance flag on the same row.

``consistency`` a contact sheet of many views: the same object keeps the same
                colour across viewpoints, which is the novel-view consistency
                evidence.  (Just a relabelled crop of the overlay directory.)

Usage
-----
    PYTHONNOUSERSITE=1 python scripts/make_ticket05_figures.py \
        --ours_overlay  <out>/overlay --baseline_overlay <t03>/overlay_bench_default \
        --table <out>/instance_physics.json --names_json names.json \
        --out_dir <figures>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/"
                "site-packages/matplotlib/mpl-data/fonts/ttf")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf") if mono else \
           ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def banner(width: int, text: str, h: int = 54, fill=(24, 24, 24),
           colour=(255, 255, 255), size: int = 30) -> Image.Image:
    img = Image.new("RGB", (width, h), fill)
    f = font(size, bold=True)
    while size > 9 and f.getlength(text) > width - 28:   # never let a caption clip
        size -= 1
        f = font(size, bold=True)
    ImageDraw.Draw(img).text((14, h // 2), text, font=f, fill=colour, anchor="lm")
    return img


def stack(imgs: list[Image.Image]) -> Image.Image:
    w = max(i.width for i in imgs)
    out = Image.new("RGB", (w, sum(i.height for i in imgs)), (255, 255, 255))
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height
    return out


def hstack(imgs: list[Image.Image], gap: int = 10) -> Image.Image:
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


# --------------------------------------------------------------------------- #
def make_sidebyside(baseline_dir: Path, ours_dir: Path, views: list[str],
                    out_path: Path, scale: float = 0.55) -> dict:
    rows, stats = [], {}
    for v in views:
        b = Image.open(baseline_dir / f"{v}_overlay.png").convert("RGB")
        o = Image.open(ours_dir / f"{v}_overlay.png").convert("RGB")
        diff = int((np.asarray(b).astype(np.int16) != np.asarray(o).astype(np.int16))
                   .any(-1).sum())
        stats[v] = {"differing_pixels": diff, "total_pixels": b.width * b.height}
        w, h = int(b.width * scale), int(b.height * scale)
        b, o = b.resize((w, h), Image.LANCZOS), o.resize((w, h), Image.LANCZOS)
        pair = hstack([b, o], gap=8)
        cap = Image.new("RGB", (pair.width, 34), (255, 255, 255))
        d = ImageDraw.Draw(cap)
        d.text((6, 17), f"view {v}", font=font(20, bold=True), fill=(0, 0, 0), anchor="lm")
        d.text((pair.width - 6, 17),
               f"full-resolution differing pixels between the two panels: "
               f"{diff:,} / {stats[v]['total_pixels']:,}",
               font=font(17), fill=(90, 90, 90), anchor="rm")
        rows += [pair, cap]

    body = stack(rows)
    head = stack([
        banner(body.width, "Instance segmentation, side by side  -  same scene, "
                           "same views, same renderer"),
        banner(body.width,
               "LEFT: official IGGT weights, ticket 03 baseline      |      "
               "RIGHT: this map's pipeline (IGGT + trained physics head, one forward)",
               h=40, fill=(245, 245, 245), colour=(30, 30, 30), size=21),
    ])
    out = stack([head, body])
    out.save(out_path)
    return stats


# --------------------------------------------------------------------------- #
def make_physics_panel(overlay_png: Path, labelmap_npy: Path, table_json: Path,
                       names: dict, out_path: Path, small_threshold: int,
                       scale: float = 1.0) -> None:
    tab = json.loads(table_json.read_text())
    rows = {r["instance_id"]: r for r in tab["rows"]}
    meta = tab["meta"]
    palette = json.loads((overlay_png.parent / "overlay_summary.json").read_text())["palette"]

    img = Image.open(overlay_png).convert("RGB")
    lbl = np.load(labelmap_npy)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    sy, sx = img.height / lbl.shape[0], img.width / lbl.shape[1]

    from scipy.ndimage import distance_transform_edt

    for inst in sorted(rows):
        m = lbl == inst
        if m.sum() < 200:
            continue
        # deepest interior point: the callout must land ON its own instance, and
        # a centroid does not (the wall's centroid falls on the doll in front of it)
        dt = distance_transform_edt(m)
        py, px = np.unravel_index(int(dt.argmax()), dt.shape)
        cy, cx = int(py * sy), int(px * sx)
        col = tuple(palette[str(inst)])
        r = 21
        # a stuff instance's deepest point can sit in a corner; keep the disc whole
        cx = int(np.clip(cx, r + 4, img.width - r - 4))
        cy = int(np.clip(cy, r + 4, img.height - r - 4))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(0, 0, 0), width=3)
        d.text((cx, cy), str(inst), font=font(26, bold=True), fill=(0, 0, 0), anchor="mm")

    # ---- table ----
    hdr = ["#", "instance", "n_gau", "trust", "rho kg/m3", "E  Pa", "nu",
           "var (r,E,n)", "mu_spread (r,E,n)"]
    body = []
    for inst in sorted(rows):
        r = rows[inst]
        body.append([
            str(inst), names.get(str(inst), f"instance {inst}"), f"{r['n_gau']:,}",
            "LOW" if r["n_gau"] < small_threshold else "ok",
            f"{r['rho_si']:.1f}", f"{r['E_si']:.3e}", f"{r['nu_si']:.4f}",
            ", ".join(f"{v:.2f}" for v in r["var"]),
            ", ".join(f"{v:.2f}" for v in r["mu_spread"]),
        ])

    f = font(20, mono=True)
    fb = font(20, bold=True, mono=True)
    widths = [max(len(hdr[c]), *(len(b[c]) for b in body)) for c in range(len(hdr))]
    cw = f.getlength("0")
    pad = 18
    tab_w = max(int(sum((w + 2) * cw for w in widths)) + 2 * pad, 1180)
    tab_h = 46 * (len(body) + 4) + 150
    t = Image.new("RGB", (tab_w, tab_h), (255, 255, 255))
    td = ImageDraw.Draw(t)
    y = 16
    td.text((pad, y), "Instance physics table", font=font(28, bold=True), fill=(0, 0, 0))
    y += 42
    td.text((pad, y),
            f"physics head @ global_step={meta['phys_ckpt_global_step']}   "
            f"IGGT input {meta['image_size']}   HDBSCAN {meta['hdbscan']['min_cluster_size']}/"
            f"{meta['hdbscan']['min_samples']}/{meta['hdbscan']['cluster_selection_epsilon']}"
            f" @ max_points={meta['hdbscan']['max_points']}",
            font=font(17), fill=(70, 70, 70))
    y += 26
    for line in ["var = the head's own softplus output (model uncertainty).",
                 "mu_spread = std of mu when each Gaussian is decoded separately "
                 "(member consistency).",
                 "var and mu_spread are two different things and are never merged; "
                 "both live in z-score space, not SI."]:
        td.text((pad, y), line, font=font(17), fill=(70, 70, 70))
        y += 22
    for line in [f"trust=LOW: fewer than {small_threshold:,} Gaussians.  Ticket 04 measured "
                 "the 2D<->3D pooling cosine collapsing to 0.39-0.82",
                 "on small/thin instances, against 0.95-0.99 above ~1000 Gaussians.  "
                 "Those rows are not as trustworthy as the rest.",
                 "E is not precise to better than about a factor 1.5: ticket 06 measured the "
                 "GT-vs-HDBSCAN membership mismatch at <=0.16 z,",
                 "which decodes through 10^(1.318 z) to up to +47% in Pa.  "
                 "(The mismatch depends on WHICH Gaussians differ, not how many.)"]:
        td.text((pad, y), line, font=font(17), fill=(150, 40, 40))
        y += 22
    y += 18

    def draw_row(cells, fnt, colour=(0, 0, 0)):
        x = pad
        for c, (cell, w) in enumerate(zip(cells, widths)):
            td.text((x, y), cell, font=fnt, fill=colour)
            x += (w + 2) * cw

    draw_row(hdr, fb)
    y += 34
    td.line([(pad, y - 6), (tab_w - pad, y - 6)], fill=(0, 0, 0), width=2)
    for inst, cells in zip(sorted(rows), body):
        col = tuple(palette[str(inst)])
        td.rectangle([pad - 12, y + 4, pad - 4, y + 20], fill=col, outline=(0, 0, 0))
        draw_row(cells, f, (150, 40, 40) if cells[3] == "LOW" else (0, 0, 0))
        y += 34
    t = t.crop((0, 0, tab_w, y + 20))

    h = max(img.height, t.height)
    canvas = Image.new("RGB", (img.width + t.width + 24, h + 60), (255, 255, 255))
    canvas.paste(banner(canvas.width,
                        "3D instances, each carrying (E, nu, rho)  -  "
                        "full-resolution overlay: photo underneath, instance colours on top"),
                 (0, 0))
    canvas.paste(img, (0, 60))
    canvas.paste(t, (img.width + 24, 60))
    canvas.save(out_path)


# --------------------------------------------------------------------------- #
def make_comparison(table_a: Path, table_b: Path, names: dict, label_a: str,
                    label_b: str, out_md: Path, control: Path | None = None) -> None:
    """Two decodes of the same instances, side by side, as a markdown table.

    Used for the resolution decision: the same Gaussians and the same membership,
    decoded from an IGGT forward at two different input resolutions.  A control
    column (the same command run twice) is what says whether a difference is real.
    """
    A = {r["instance_id"]: r for r in json.loads(table_a.read_text())["rows"]}
    B = {r["instance_id"]: r for r in json.loads(table_b.read_text())["rows"]}
    lines = [f"| id | instance | n_gau | rho {label_a} | rho {label_b} | ratio | "
             f"E {label_a} | E {label_b} | ratio | nu {label_a} | nu {label_b} | delta |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fr, fe, dn = [], [], []
    for i in sorted(A):
        x, y = A[i], B[i]
        fr.append(y["rho_si"] / x["rho_si"])
        fe.append(y["E_si"] / x["E_si"])
        dn.append(y["nu_si"] - x["nu_si"])
        lines.append(f"| {i} | {names.get(str(i), i)} | {x['n_gau']:,} | "
                     f"{x['rho_si']:.1f} | {y['rho_si']:.1f} | {fr[-1]:.2f}x | "
                     f"{x['E_si']:.3e} | {y['E_si']:.3e} | {fe[-1]:.2f}x | "
                     f"{x['nu_si']:.4f} | {y['nu_si']:.4f} | {dn[-1]:+.4f} |")
    fr, fe = np.array(fr), np.array(fe)
    lines += ["", f"rho ratio: median {np.median(fr):.2f}x, range "
                  f"{fr.min():.2f}x-{fr.max():.2f}x",
              f"E   ratio: median {np.median(fe):.2f}x, range {fe.min():.2f}x-{fe.max():.2f}x",
              f"nu  |delta|: median {np.median(np.abs(dn)):.4f}, "
              f"max {np.abs(dn).max():.4f}"]
    for tag, d in ((label_a, A), (label_b, B)):
        v = np.array([d[i]["var"] for i in sorted(d)])
        sp = np.array([d[i]["mu_spread"] for i in sorted(d)])
        lines.append(f"{tag}: mean var (rho,E,nu) = {v.mean(0).round(4).tolist()}   "
                     f"mean mu_spread = {sp.mean(0).round(4).tolist()}")
    if control is not None:
        C = {r["instance_id"]: r for r in json.loads(control.read_text())["rows"]}
        cr = max(abs(C[i]["rho_si"] / A[i]["rho_si"] - 1) for i in sorted(A))
        ce = max(abs(C[i]["E_si"] / A[i]["E_si"] - 1) for i in sorted(A))
        cn = max(abs(C[i]["nu_si"] - A[i]["nu_si"]) for i in sorted(A))
        lines += ["", "CONTROL (the same command run twice; nothing below this is signal):",
                  f"  rho max |1-ratio| = {cr:.3e}   E max |1-ratio| = {ce:.3e}   "
                  f"nu max |delta| = {cn:.3e}"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[compare] -> {out_md}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours_overlay", required=True)
    ap.add_argument("--baseline_overlay", default=None)
    ap.add_argument("--table", required=True, help="instance_physics.json")
    ap.add_argument("--names_json", default=None,
                    help='{"1": "cat figurine", ...} for the table')
    ap.add_argument("--views", default="00,12,24")
    ap.add_argument("--physics_view", default="00")
    ap.add_argument("--small_threshold", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--compare_table", default=None,
                    help="a second instance_physics.json decoded from the same "
                         "membership at another operating point")
    ap.add_argument("--compare_labels", default="A,B")
    ap.add_argument("--control_table", default=None,
                    help="instance_physics.json from re-running the SAME command "
                         "(the noise floor of the comparison)")
    args = ap.parse_args()

    ours = Path(args.ours_overlay)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    names = json.loads(Path(args.names_json).read_text()) if args.names_json else {}

    if args.baseline_overlay:
        stats = make_sidebyside(Path(args.baseline_overlay), ours, views,
                                out / "sidebyside_baseline_vs_ours.png")
        (out / "sidebyside_pixel_diff.json").write_text(json.dumps(stats, indent=1))
        print("[sidebyside] full-resolution differing pixels per view:")
        for v, s in stats.items():
            print(f"    view {v}: {s['differing_pixels']:,} / {s['total_pixels']:,}")

    if args.compare_table:
        la, lb = [x.strip() for x in args.compare_labels.split(",")]
        make_comparison(Path(args.table), Path(args.compare_table), names, la, lb,
                        out / "resolution_comparison.md",
                        Path(args.control_table) if args.control_table else None)

    make_physics_panel(ours / f"{args.physics_view}_overlay.png",
                       ours / f"{args.physics_view}_labelmap.npy",
                       Path(args.table), names,
                       out / "instance_physics_panel.png", args.small_threshold)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()

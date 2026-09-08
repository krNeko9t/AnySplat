"""04 号票：跨批 query 池化 + 3D 高斯集合 IoU 去重。

输入是 ``trace_instance_to_gaussians.py --feature_source segvggt`` 落盘的 ``.pt``：
全局的 ``feat [P,128]``（trace 出来的 128 维 feature field）和 ``query_bank``
（M = Σ_b N_b 个跨批候选，每行自带 128 维投影、score、2D mask 面积、和它那一行物性）。

一批 4 视角只含那 4 个视角里的物体，所以单批 query 不覆盖场景（地图核心论证）。
做法是把所有批并成 ``Q_all``，与**全局**的 field 做点积拿到每个候选在全部 P 个高斯上的
响应，再在 3D 里按高斯集合 IoU 合并 —— 不按 query embedding 的余弦合并，那等于又信
了一遍跨批一致性，而 IoU 是在 3D 里直接看重叠。

输出：实例表（id / 高斯数 / score / 成员数 / E,ν,ρ）、逐高斯实例标签、上色 PLY。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def gaussian_logits(feat: torch.Tensor, q_proj: torch.Tensor,
                    chunk: int = 200_000) -> torch.Tensor:
    """``feat @ q_proj.T`` -> [P, M], chunked over Gaussians to bound memory."""
    out = torch.empty(feat.shape[0], q_proj.shape[0],
                      device=feat.device, dtype=torch.float32)
    for s in range(0, feat.shape[0], chunk):
        e = min(s + chunk, feat.shape[0])
        out[s:e] = feat[s:e] @ q_proj.T
    return out


def membership(logits: torch.Tensor, valid: torch.Tensor,
               thr: float) -> torch.Tensor:
    """Soft (overlapping) per-candidate Gaussian sets: ``logit > thr``.

    Deliberately not an ``argmax`` over candidates: a hard partition would force
    every Gaussian to pick an owner, and the size of the *unclaimed* remainder is
    itself a number this route has to report rather than define away.

    ``thr = 0`` is the model's own 2D operating point carried over, not a tuned
    number: on the 2D masks the decision boundary sits at ``cos(q, f) ~= 0``
    (positives p05 = 0.000, negatives p99 = -0.017), and trace is a positive-weighted
    average so it rescales ``|f|`` (2D p50 = 3.46 -> 3D p50 = 0.043) without moving
    its direction. The sign of ``q . f`` is what survives the rescaling.
    """
    return (logits > thr) & valid[:, None]


# ---------------------------------------------------------------------------
# 3D dedup
# ---------------------------------------------------------------------------

def pairwise_iou(sets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian-set IoU and containment over candidates. ``sets`` is [P, M] bool.

    Returns ``(iou [M, M], containment [M, M])`` where ``containment[i, j]`` is
    ``|S_i & S_j| / |S_i|`` -- a part fully inside a whole scores low on IoU but
    high here, so the two are reported side by side.
    """
    f = sets.float()
    inter = f.T @ f
    size = f.sum(0)
    union = size[:, None] + size[None, :] - inter
    return inter / union.clamp(min=1.0), inter / size[:, None].clamp(min=1.0)


def greedy_merge(iou: torch.Tensor, scores: torch.Tensor,
                 thr: float) -> list[list[int]]:
    """Score-ordered greedy merge. Returns member lists, representative first.

    Standard NMS shape, but the loser is *absorbed* rather than discarded: the same
    physical object is re-proposed once per batch that saw it, and those duplicates
    carry that object's other physics readouts, which the caller pools.
    """
    order = scores.argsort(descending=True).tolist()
    groups: list[list[int]] = []
    for i in order:
        for g in groups:
            if iou[i, g[0]] > thr:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def pool_physics(mu_model: torch.Tensor, var_model: torch.Tensor,
                 scores: torch.Tensor, members: list[int]):
    """Pool a merged group's physics readouts.

    The mean is taken in **model space** and denormalised afterwards, because
    ``physgm_denormalize`` is ``10 ** (x * std + mean)`` for density and E: the model
    space is the log domain, which is where an average of several estimates of the
    same object belongs (a score-weighted geometric mean in SI).

    ``var`` is the *predicted* variance of the readout head, not a regression of any
    GT spread (see ``query_physgm_readout.py``), so it is pooled as a plain weighted
    mean and reported separately from ``mu_spread`` -- the std of the members' own mu,
    which is the honest "do the batches agree" number. Ticket 06 owns how these are
    presented; this only keeps them apart.
    """
    idx = torch.tensor(members)
    w = scores[idx].clamp(min=1e-6)
    w = w / w.sum()
    mu = (mu_model[idx] * w[:, None]).sum(0)
    var = (var_model[idx] * w[:, None]).sum(0)
    spread = mu_model[idx].std(0) if len(members) > 1 else torch.zeros_like(mu)
    return mu, var, spread


def render_instances(inst_ply, orig_ply, source_path, backend, out_dir, args):
    """Render original / instance-coloured pairs from a few scene views."""
    import random

    from PIL import Image
    import trace_instance_to_gaussians as T
    from src.trace_cameras import load_trace_cameras
    from src.trace_render.trace_rasterize import (
        TRACE_CHANNELS, load_gaussians_from_ply, resolve_trace_backend,
        trace_single_view,
    )

    if not source_path or not os.path.isdir(source_path):
        print(f"[pool] no scene dir at {source_path!r}; skipping renders")
        return
    cams = load_trace_cameras("colmap", source_path=os.path.abspath(source_path),
                              images_folder="images",
                              resolution=args.render_resolution)
    random.seed(0)
    sel = random.sample(cams, min(args.render_views, len(cams)))
    rdir = out_dir / "renders"
    rdir.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor([1.0, 1.0, 1.0], device=args.device)

    frames = {}
    for tag, path in (("original", orig_ply), ("instances", inst_ply)):
        means, quats, scales, opacities, colors = load_gaussians_from_ply(path)
        rb = backend or resolve_trace_backend("auto", scales.shape[1])
        for cam in sel:
            H, W = cam.image_height, cam.image_width
            img_sem = torch.zeros(H, W, TRACE_CHANNELS, device=args.device)
            img_mask = torch.ones(H, W, dtype=torch.int32, device=args.device)
            with torch.no_grad():
                *_, out_color = trace_single_view(means, quats, scales, opacities,
                                                  colors, img_sem, img_mask, cam, bg, rb)
            arr = (out_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
            frames.setdefault(cam.image_name, {})[tag] = arr.astype(np.uint8)

    for name, pair in frames.items():
        Image.fromarray(np.concatenate([pair["original"], pair["instances"]], axis=1)
                        ).save(rdir / f"{name}_compare.png")
    print(f"[pool] wrote {len(frames)} comparison renders to {rdir}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traced", required=True,
                    help="gaussian_segvggt_feat.pt from the segvggt feature source")
    ap.add_argument("--ply_path", default=None,
                    help="scene PLY to recolour (default: the one recorded in the .pt)")
    ap.add_argument("-o", "--output_dir", default=None,
                    help="default: <traced dir>/instances")
    ap.add_argument("--logit_thr", type=float, default=0.0,
                    help="Gaussian membership threshold on q.f (default: the model's "
                         "own 2D boundary, see membership())")
    ap.add_argument("--iou_thr", type=float, default=0.3,
                    help="merge two candidates whose Gaussian sets overlap above this")
    ap.add_argument("--max_mask_frac", type=float, default=0.3,
                    help="drop candidates whose 2D mask covers more than this fraction "
                         "of the frame -- the background/'stuff' slot. On bench the "
                         "background slot sits at 0.49-0.77 and every real object at "
                         "<= 0.126, so the default falls in a 4x-wide empty gap.")
    ap.add_argument("--max_scene_frac", type=float, default=0.2,
                    help="3D backstop for the same thing: drop candidates claiming more "
                         "than this fraction of the traced Gaussians. Applies where the "
                         "2D area filter misses (bench batch 2 reads 0.492).")
    ap.add_argument("--min_gaussians", type=int, default=200,
                    help="drop candidates claiming fewer Gaussians than this "
                         "(the 3D counterpart of decode_instances' npoint_thr)")
    ap.add_argument("--min_num_ray", type=int, default=1,
                    help="a Gaussian must be hit by at least this many rays to be "
                         "assignable (default 1 = 'was traced at all')")
    ap.add_argument("--render_views", type=int, default=3,
                    help="render this many scene views of the coloured instances "
                         "(0 to skip); the qualitative figure this ticket asks for")
    ap.add_argument("--render_resolution", type=int, default=4)
    ap.add_argument("--source_path", default=None,
                    help="scene dir for the render cameras (default: from the .pt)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir or (Path(args.traced).parent / "instances"))
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(args.traced, map_location="cpu", weights_only=False)
    bank = d.get("query_bank")
    if bank is None:
        raise SystemExit(f"{args.traced} has no query_bank -- was it traced with "
                         "--feature_source segvggt?")
    dev = args.device
    feat = d["feat"].float().to(dev)
    num_ray = d["num_ray"].to(dev)
    P = feat.shape[0]

    q_proj = bank["query_proj"].float().to(dev)
    scores = bank["scores"].float()
    mu_model = bank["phys_mu_model"].float()
    var_model = bank["phys_var_model"].float()
    batch_id = bank["batch_id"]
    query_idx = bank["query_idx"]
    names = list(bank["property_names"])
    M0 = q_proj.shape[0]

    print(f"[pool] {P:,} Gaussians x {M0} cross-batch candidates "
          f"over {int(batch_id.max()) + 1} batches")

    # ---- 1. drop the "stuff" candidates (2D area), see membership() docstring ----
    mask_frac = bank.get("mask_frac")
    if mask_frac is None:
        print("[pool] WARNING: this .pt predates the mask_frac field; the background "
              "slot cannot be filtered in 2D and will swallow the scene in 3D. "
              "Re-run the trace to get it.")
        keep_2d = torch.ones(M0, dtype=torch.bool)
    else:
        keep_2d = mask_frac <= args.max_mask_frac
        drop = (~keep_2d).nonzero().flatten()
        print(f"[pool] 2D area filter (mask_frac <= {args.max_mask_frac}): "
              f"dropped {len(drop)} of {M0}")
        for i in drop.tolist():
            print(f"         batch {int(batch_id[i])} slot {int(query_idx[i])}: "
                  f"{100 * mask_frac[i]:.1f}% of frame, score {scores[i]:.2f}")

    # ---- 2. per-candidate Gaussian sets ----
    valid = num_ray >= args.min_num_ray
    print(f"[pool] assignable Gaussians (num_ray >= {args.min_num_ray}): "
          f"{int(valid.sum()):,} ({100 * valid.float().mean():.1f}%)")

    logits = gaussian_logits(feat, q_proj)
    sets = membership(logits, valid, args.logit_thr)

    sizes = sets.sum(0).cpu()
    n_valid = int(valid.sum())
    not_stuff = sizes <= args.max_scene_frac * n_valid
    n_stuff = int((keep_2d & ~not_stuff).sum())
    if n_stuff:
        for i in (keep_2d & ~not_stuff).nonzero().flatten().tolist():
            print(f"[pool] 3D area backstop: batch {int(batch_id[i])} slot "
                  f"{int(query_idx[i])} claims {100 * sizes[i] / n_valid:.1f}% of the "
                  f"traced Gaussians (2D mask_frac "
                  f"{float(mask_frac[i]) if mask_frac is not None else float('nan'):.3f})")
    keep = keep_2d & not_stuff & (sizes >= args.min_gaussians)
    n_small = int((keep_2d & not_stuff & (sizes < args.min_gaussians)).sum())
    print(f"[pool] size filter (>= {args.min_gaussians} Gaussians): dropped {n_small}")
    kidx = keep.nonzero().flatten()
    if len(kidx) == 0:
        raise SystemExit("[pool] no candidate survived filtering")
    print(f"[pool] {len(kidx)} candidates survive; claimed Gaussians "
          f"{int(sets[:, kidx.to(dev)].any(1).sum()):,} "
          f"({100 * sets[:, kidx.to(dev)].any(1).float().mean():.1f}%)")

    sets_k = sets[:, kidx.to(dev)]
    logits_k = logits[:, kidx.to(dev)]

    # ---- 3. merge in 3D by Gaussian-set IoU ----
    iou, cont = pairwise_iou(sets_k)
    iou_c, cont_c = iou.cpu(), cont.cpu()
    off = ~torch.eye(len(kidx), dtype=torch.bool)
    print(f"[pool] pairwise IoU: p50={iou_c[off].median():.3f} "
          f"p90={iou_c[off].quantile(0.9):.3f} max={iou_c[off].max():.3f}; "
          f"containment>0.8 pairs={int((cont_c > 0.8)[off].sum())}")

    groups = greedy_merge(iou_c, scores[kidx], args.iou_thr)
    print(f"[pool] merged {len(kidx)} candidates -> {len(groups)} instances "
          f"(IoU > {args.iou_thr})")

    # ---- 4. per-Gaussian labels (argmax over claiming instances, viz only) ----
    rep = torch.tensor([g[0] for g in groups])
    inst_logits = logits_k[:, rep.to(dev)]
    inst_sets = sets_k[:, rep.to(dev)]
    best = inst_logits.masked_fill(~inst_sets, float("-inf")).argmax(1)
    labels = torch.where(inst_sets.any(1), best, torch.full_like(best, -1)).cpu()

    # An instance can survive the per-candidate size filter and still lose every one
    # of its Gaussians to a higher-scoring overlapping instance at argmax time. Drop
    # those here rather than reporting a row with no geometry behind it.
    counts = torch.bincount(labels[labels >= 0], minlength=len(groups))
    alive = (counts >= args.min_gaussians).nonzero().flatten()
    if len(alive) < len(groups):
        print(f"[pool] dropped {len(groups) - len(alive)} instances left with "
              f"< {args.min_gaussians} Gaussians after argmax "
              f"(sizes {counts[counts < args.min_gaussians].tolist()})")
        remap = torch.full((len(groups),), -1, dtype=labels.dtype)
        remap[alive] = torch.arange(len(alive), dtype=labels.dtype)
        labels = torch.where(labels >= 0, remap[labels.clamp(min=0)], labels)
        groups = [groups[i] for i in alive.tolist()]
        inst_sets = inst_sets[:, alive.to(dev)]
    n_claim = int((labels >= 0).sum())
    print(f"[pool] labelled {n_claim:,} Gaussians ({100 * n_claim / P:.1f}%); "
          f"unclaimed {P - n_claim:,} ({100 * (P - n_claim) / P:.1f}%)")

    # ---- 5. instance table ----
    from src.dataset.physics.parsers import physgm_denormalize

    rows = []
    for gi, g in enumerate(groups):
        mu, var, spread = pool_physics(mu_model[kidx], var_model[kidx], scores[kidx], g)
        si = physgm_denormalize(mu[None])[0]
        si_hi = physgm_denormalize((mu + spread)[None])[0]
        rows.append({
            "instance": gi,
            "n_gaussians": int((labels == gi).sum()),
            "n_gaussians_claimed": int(inst_sets[:, gi].sum()),
            "score": float(scores[kidx][g[0]]),
            "n_members": len(g),
            "n_batches": int(len(set(batch_id[kidx][torch.tensor(g)].tolist()))),
            "slots": sorted(set(int(x) for x in query_idx[kidx][torch.tensor(g)])),
            **{n: float(si[i]) for i, n in enumerate(names)},
            **{f"{n}_hi_1sigma": float(si_hi[i]) for i, n in enumerate(names)},
            **{f"{n}_var_model": float(var[i]) for i, n in enumerate(names)},
        })
    rows.sort(key=lambda r: -r["n_gaussians"])

    hdr = (f"{'id':>3} {'#gauss':>8} {'#claim':>8} {'score':>6} {'mem':>4} "
           f"{'bat':>4} " + " ".join(f"{n:>12}" for n in names))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['instance']:>3} {r['n_gaussians']:>8,} {r['n_gaussians_claimed']:>8,} "
            f"{r['score']:>6.2f} {r['n_members']:>4} {r['n_batches']:>4} "
            + " ".join(f"{r[n]:>12.4g}" for n in names))
    table = "\n".join(lines)
    print("\n[pool] instances (physics in SI, pooled over members):\n" + table)
    (out_dir / "instances.txt").write_text(table + "\n")
    (out_dir / "instances.json").write_text(json.dumps(rows, indent=2))

    np.savez(
        out_dir / "instances.npz",
        labels=labels.numpy().astype(np.int32),
        n_instances=len(groups),
        candidate_index=kidx.numpy(),
        member_groups=np.array([json.dumps(g) for g in groups]),
        params=json.dumps(vars(args)),
    )

    # ---- 6. coloured PLY ----
    import trace_instance_to_gaussians as T

    ply_path = args.ply_path or d.get("ply_path")
    if ply_path and os.path.isfile(ply_path):
        rng = np.random.default_rng(0)
        palette = rng.integers(40, 255, size=(len(groups), 3), dtype=np.uint8)
        rgb = np.full((P, 3), 110, dtype=np.uint8)          # unclaimed: grey
        lab = labels.numpy()
        rgb[lab >= 0] = palette[lab[lab >= 0]]
        T.save_colored_gaussians_ply(ply_path, rgb, str(out_dir / "instances.ply"))
        if args.render_views > 0:
            render_instances(str(out_dir / "instances.ply"), ply_path,
                             args.source_path or d.get("source_path"),
                             d.get("trace_backend"), out_dir, args)
    else:
        print(f"[pool] no PLY at {ply_path!r}; skipping the coloured export")

    print(f"[pool] wrote {out_dir}")


if __name__ == "__main__":
    main()

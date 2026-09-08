"""SegVGGT end-to-end instance-segmentation inference (iteration 1).

Loads the official SegVGGT checkpoint (HuggingFace ``JinyuanQu/SegVGGT``) into the
repo's ``SegVGGTModel`` and runs the object-query instance branch on a folder of
multi-view images, producing per-view instance overlays. No clustering, no GT masks:
each learnable query directly yields a per-view mask + a class distribution whose last
channel is *no-match* (unmatched / empty slot; DETR: no-object); queries surviving
score / area thresholds become instances.

Preprocessing matches the official eval exactly (width -> 518, height rounded to a
multiple of 14, portrait center-cropped square, pixels in [0, 1]).

Decoding mirrors ``eval/instance_eval_common.predict_by_feat_instance`` (class-aware
by default). A non-zero ``--score_thr`` is required for readable overlays: with the
old default of 0.0 nearly every query survives and ~300 overlapping masks look like
noise.

Examples:
  # official ScanNetv2 weights (20 semantic, exclude wall/floor -> 18+1 head)
  python scripts/segvggt_infer.py --hf --num_semantic_classes 20 \
      --image_dir examples/vrnerf/riverview --out_dir outputs/segvggt_demo

  # local checkpoint, ScanNet200 head
  python scripts/segvggt_infer.py --ckpt /path/segvggt_scannet200.pt \
      --num_semantic_classes 200 \
      --image_dir /data/scene/color --max_views 12 --out_dir outputs/seg
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.arch import get_model
from src.model.arch.segvggt import EncoderSegVGGTCfg
from src.model.arch.segvggt_decode import decode_instances, report_physics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("segvggt_infer")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------------------------------------------------------------------------- #
# preprocessing (mirrors segvggt/eval/eval_instance_seg.py load_single_scene_data)
# --------------------------------------------------------------------------- #
def load_and_preprocess(paths: list[Path], target_width: int = 518) -> torch.Tensor:
    """Return image tensor [1, S, 3, H, W] in [0, 1]."""
    imgs = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("skip unreadable image %s", p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        new_w = target_width
        new_h = round(h * (new_w / w) / 14) * 14
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        if new_h > new_w:  # portrait -> center square crop
            y0 = (new_h - new_w) // 2
            rgb = rgb[y0 : y0 + new_w, :, :]
        imgs.append(rgb)
    if not imgs:
        raise RuntimeError("no readable images")
    # frames may differ in height across a folder; enforce a common H (min) by crop
    min_h = min(im.shape[0] for im in imgs)
    min_h = (min_h // 14) * 14
    imgs = [im[:min_h, :, :] for im in imgs]
    arr = np.stack(imgs, axis=0)  # [S, H, W, 3] uint8
    t = torch.from_numpy(arr[None]).float().div_(255.0)  # [1, S, H, W, 3]
    return t.permute(0, 1, 4, 2, 3).contiguous()  # [1, S, 3, H, W]


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(40, 230, size=(max(n, 1), 3), dtype=np.uint8)


def visualize(
    images: torch.Tensor,
    binary: torch.Tensor,
    V: int,
    h: int,
    w: int,
    out_dir: Path,
    alpha: float = 0.5,
    max_instances: int = 50,
):
    """Overlay per-view instance masks and write PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if binary.numel() == 0:
        logger.warning("no instances to visualize")
        return
    if binary.shape[0] > max_instances:
        logger.info(
            "visualizing top-%d / %d instances (raise --max_instances to show more)",
            max_instances,
            binary.shape[0],
        )
        binary = binary[:max_instances]

    imgs = (images[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # [V,H,W,3]
    H, W = imgs.shape[1:3]
    colors = _palette(binary.shape[0])
    masks = binary.view(binary.shape[0], V, h, w).cpu().numpy()  # [N,V,h,w]

    for v in range(V):
        base = imgs[v].copy()
        overlay = base.copy()
        for i in range(masks.shape[0]):
            mk = masks[i, v].astype(np.uint8)
            if mk.sum() == 0:
                continue
            mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[mk] = colors[i]
        blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
        cv2.imwrite(
            str(out_dir / f"view_{v:03d}_instances.png"),
            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
        )
    logger.info("wrote %d view overlays -> %s", V, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="folder of multi-view images")
    ap.add_argument(
        "--ckpt",
        default="",
        help="local checkpoint -- either an official SegVGGT .pt, or a fine-tuned "
             "Lightning .ckpt from SegVGGTWrapper (its 'model.' key prefix is "
             "stripped automatically)",
    )
    ap.add_argument("--hf", action="store_true",
                    help="download the checkpoint from JinyuanQu/SegVGGT")
    ap.add_argument(
        "--num_semantic_classes",
        type=int,
        default=20,
        choices=[20, 200],
        help="ScanNet semantic table size (20=scannetv2.pt, 200=scannet200.pt)",
    )
    ap.add_argument(
        "--non_instance_classes",
        default="wall,floor",
        help="comma-separated semantic class names excluded from the instance head "
             "(default: wall,floor). Empty string = exclude none",
    )
    ap.add_argument("--max_views", type=int, default=8)
    ap.add_argument("--mask_thr", type=float, default=0.4)
    ap.add_argument(
        "--score_thr",
        type=float,
        default=0.25,
        help="drop queries with score <= thr (0.0 keeps ~all and yields noisy overlays)",
    )
    ap.add_argument("--npoint_thr", type=int, default=200)
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument(
        "--class_agnostic",
        action="store_true",
        help="score by 1-P(no-match); default is official class-aware top-k",
    )
    ap.add_argument("--max_instances", type=int, default=50,
                    help="max instances to draw in the overlay")
    ap.add_argument(
        "--phys_scheme",
        default=None,
        choices=["query_physgm"],
        help="attach the per-query physics readout (required to load a "
             "phys-on-query checkpoint; without it from_checkpoint refuses one)",
    )
    ap.add_argument("--out_dir", default="outputs/segvggt_demo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = args.ckpt
    if args.hf and not ckpt:
        from huggingface_hub import hf_hub_download
        fname = (
            f"checkpoint/segvggt_scannet"
            f"{'v2' if args.num_semantic_classes == 20 else '200'}.pt"
        )
        logger.info("downloading %s from JinyuanQu/SegVGGT ...", fname)
        ckpt = hf_hub_download("JinyuanQu/SegVGGT", fname)
    if not ckpt:
        raise SystemExit("provide --ckpt PATH or --hf")

    paths = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix in IMG_EXTS)
    if args.max_views:
        paths = paths[: args.max_views]
    logger.info("using %d views from %s", len(paths), args.image_dir)
    images = load_and_preprocess(paths).to(args.device)

    non_inst = [s.strip() for s in args.non_instance_classes.split(",") if s.strip()]
    cfg = EncoderSegVGGTCfg(
        name="segvggt",
        num_semantic_classes=args.num_semantic_classes,
        non_instance_classes=non_inst,
        pretrained_weights=ckpt,
        phys_scheme=args.phys_scheme,
    )
    model = get_model(cfg).to(args.device).eval()

    with torch.no_grad():
        enc_out, _ = model(images)
    pred = enc_out.segvggt_prediction
    qm = pred.query_masks[0]          # [Q, V, h, w]
    ql = pred.query_class_logits[0]   # [Q, C+1]
    Q, V, h, w = qm.shape
    logger.info("queries=%d views=%d mask=%dx%d classes=%d", Q, V, h, w, ql.shape[-1])

    binary, scores, labels, query_idx = decode_instances(
        ql,
        qm.reshape(Q, -1),
        mask_thr=args.mask_thr,
        topk=args.topk,
        npoint_thr=args.npoint_thr,
        score_thr=args.score_thr,
        class_agnostic=args.class_agnostic,
    )
    logger.info(
        "kept %d instances (scores %.3f..%.3f, mode=%s)",
        binary.shape[0],
        float(scores.max()) if len(scores) else 0.0,
        float(scores.min()) if len(scores) else 0.0,
        "class_agnostic" if args.class_agnostic else "class_aware",
    )
    if len(labels):
        uniq, cnt = labels.unique(return_counts=True)
        logger.info(
            "label hist: %s",
            {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())},
        )
    report_physics(pred, query_idx, scores, Path(args.out_dir))
    visualize(
        images, binary, V, h, w, Path(args.out_dir), max_instances=args.max_instances
    )


if __name__ == "__main__":
    main()

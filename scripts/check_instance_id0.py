"""Decide what instance id 0 *means* in a manifest dataset: background, or unlabelled?

This matters for set-prediction losses (SegVGGT) and not at all for contrastive ones
(IGGT's disc/mvc).  A contrastive loss simply skips id-0 pixels, so their meaning is
irrelevant.  A set-prediction loss supervises every matched query's mask over the whole
frame, so id-0 pixels become *negatives* -- which is correct if id 0 is genuinely
background, and actively harmful if it hides unannotated objects (the model is taught
to suppress exactly the objects you wanted it to find).

The discriminator is the shape of the id-0 region, not its size:

  * true background (wall / floor / sky) -> ONE huge connected component that touches
    the image border and wraps around everything else;
  * unlabelled objects -> several compact components sitting in the *interior*, with
    object-like areas and low elongation.

So we report, per source, the fraction of id-0 pixels living in interior components,
which is the number to act on.  Rule of thumb printed at the end.

Usage:
    python scripts/check_instance_id0.py --root <DATA_ROOT> --manifest <manifest.jsonl> \
        [--scenes 60] [--frames-per-scene 3] [--label infinigen]

Run it once per source (infinigen / scannetpp_v2 / re10k) -- they are preprocessed by
different pipelines and there is no reason for them to agree.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    raise SystemExit("needs scipy (scipy.ndimage.label); pip install scipy")


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        from PIL import Image

        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def read_scenes(manifest: Path) -> list[dict]:
    if manifest.suffix == ".jsonl":
        return [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    data = json.loads(manifest.read_text())
    return data["scenes"] if isinstance(data, dict) else data


def analyse(mask: np.ndarray, min_frac: float = 1e-4) -> dict:
    """Connected-component breakdown of the id==0 region of one frame."""
    h, w = mask.shape
    total = float(h * w)
    zero = mask == 0
    out = {
        "zero_frac": zero.sum() / total,
        "n_instances": int(len(np.unique(mask[~zero]))),
        "interior_frac": 0.0,
        "n_interior_blobs": 0,
        "biggest_blob_frac": 0.0,
    }
    if not zero.any():
        return out

    lab, n = ndimage.label(zero)
    if n == 0:
        return out
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    out["biggest_blob_frac"] = float(sizes.max()) / total

    # a component is "interior" if it never touches the image border
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    border_ids = set(np.unique(border).tolist()) - {0}
    interior_px = 0
    n_blobs = 0
    for i, sz in enumerate(sizes, start=1):
        if i in border_ids:
            continue
        if sz / total < min_frac:  # speckle / antialiasing, not an object
            continue
        interior_px += int(sz)
        n_blobs += 1
    out["interior_frac"] = interior_px / total
    out["n_interior_blobs"] = n_blobs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames-per-scene", type=int, default=3)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenes = read_scenes(args.manifest)
    rng.shuffle(scenes)

    rows: list[dict] = []
    for scene in scenes[: args.scenes]:
        frames = scene.get("frames", [])
        if not frames:
            continue
        for fr in rng.sample(frames, min(args.frames_per_scene, len(frames))):
            p = args.root / fr["instance_mask_path"]
            if not p.exists():
                continue
            try:
                rows.append(analyse(load_mask(p)))
            except Exception as e:  # keep going; report at the end
                print(f"  !! {p}: {e}")

    if not rows:
        raise SystemExit("no masks read -- check --root / --manifest")

    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    name = args.label or args.manifest.parent.name
    print(f"\n===== {name}  ({len(rows)} frames from {args.scenes} scenes) =====")
    print(f"  instances per frame      : mean {col('n_instances').mean():6.2f}   "
          f"max {int(col('n_instances').max())}")
    print(f"  id==0 pixel fraction     : mean {col('zero_frac').mean():6.3f}   "
          f"p90 {np.percentile(col('zero_frac'), 90):.3f}")
    print(f"  biggest id-0 blob        : mean {col('biggest_blob_frac').mean():6.3f} "
          f"of frame  <- one big border blob => background-like")
    print(f"  INTERIOR id-0 fraction   : mean {col('interior_frac').mean():6.3f}   "
          f"p90 {np.percentile(col('interior_frac'), 90):.3f}")
    print(f"  interior id-0 blobs/frame: mean {col('n_interior_blobs').mean():6.2f}   "
          f"frames with >=1: {(col('n_interior_blobs') >= 1).mean():.0%}")

    interior = col("interior_frac").mean()
    print("\n  verdict:")
    if interior < 0.02:
        print("    id 0 behaves like TRUE BACKGROUND -> keep unlabeled_as_ignore: false")
    elif interior < 0.08:
        print("    borderline. Eyeball a few frames before deciding; the safer setting")
        print("    is unlabeled_as_ignore: false unless recall of odd objects matters.")
    else:
        print("    id 0 hides a LOT of interior structure -> likely unannotated objects.")
        print("    Set loss.segvggt.unlabeled_as_ignore: true, and watch mask area:")
        print("    with few negatives left, masks tend to over-grow.")
    print()


if __name__ == "__main__":
    main()

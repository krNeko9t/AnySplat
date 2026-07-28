"""Check I1: do depth-invalid pixels still carry non-zero instance ids?

I1 claims dataset ``valid_mask`` (depth >0 & finite & <invalid) is later reused as
``instance_valid_mask``, zeroing GT instance ids in depth holes.

make_manifest_* only gates on **file existence** (rgb+mask, optionally depth path),
not per-pixel depth validity — so holes can still enter training.

This script measures, on real ScanNet++-style frames:

  * depth-invalid pixel fraction
  * among those, how many have instance_id != 0  (would be wiped by LossSegVGGT)
  * per-instance area lost to depth holes

Usage:
    python scripts/check_i1_valid_mask_vs_instance.py \
        --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/processed_scannetpp_v2 \
        --manifest manifests/manifest_phys_scannet100.jsonl \
        --scenes 40 --frames-per-scene 4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_INVALID_VALUE = 1e9  # matches DatasetManifestCfg.depth_invalid_value


def read_scenes(manifest: Path) -> list[dict]:
    if manifest.suffix == ".jsonl":
        return [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    data = json.loads(manifest.read_text())
    return data["scenes"] if isinstance(data, dict) else data


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_inst(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def depth_valid_mask(depth: np.ndarray) -> np.ndarray:
    """Same formula as DatasetManifest.load_stack."""
    return (depth > 0) & np.isfinite(depth) & (depth < DEPTH_INVALID_VALUE)


def analyse_frame(depth: np.ndarray, inst: np.ndarray) -> dict | None:
    if depth.shape != inst.shape:
        # nearest resize inst to depth? prefer align to RGB-sized inst; resize depth
        from PIL import Image as _Image

        d = _Image.fromarray(depth)
        d = d.resize((inst.shape[1], inst.shape[0]), resample=_Image.NEAREST)
        depth = np.array(d, dtype=np.float32)

    valid = depth_valid_mask(depth)
    invalid = ~valid
    n = int(inst.size)
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return {
            "invalid_frac": 0.0,
            "polluted_frac_of_image": 0.0,
            "polluted_frac_of_invalid": 0.0,
            "inst_area": int((inst != 0).sum()),
            "inst_area_lost": 0,
            "inst_area_lost_frac": 0.0,
            "n_ids_touched": 0,
            "max_id_lost_frac": 0.0,
        }

    polluted = invalid & (inst != 0)
    n_polluted = int(polluted.sum())
    inst_fg = inst != 0
    inst_area = int(inst_fg.sum())
    inst_area_lost = int((inst_fg & invalid).sum())

    max_lost = 0.0
    n_touched = 0
    for iid in np.unique(inst[inst_fg]):
        area = int((inst == iid).sum())
        lost = int(((inst == iid) & invalid).sum())
        if lost > 0:
            n_touched += 1
            max_lost = max(max_lost, lost / area)

    return {
        "invalid_frac": n_invalid / n,
        "polluted_frac_of_image": n_polluted / n,
        "polluted_frac_of_invalid": n_polluted / n_invalid,
        "inst_area": inst_area,
        "inst_area_lost": inst_area_lost,
        "inst_area_lost_frac": (inst_area_lost / inst_area) if inst_area else 0.0,
        "n_ids_touched": n_touched,
        "max_id_lost_frac": max_lost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--frames-per-scene", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenes = read_scenes(args.manifest)
    with_depth = [
        s
        for s in scenes
        if any(fr.get("depth_path") for fr in s.get("frames", []))
    ]
    print(
        f"manifest scenes={len(scenes)} with_any_depth={len(with_depth)} "
        f"sampling up to {args.scenes} scenes x {args.frames_per_scene} frames"
    )
    if not with_depth:
        raise SystemExit("No scenes with depth_path — I1 path cannot fire (RE10K-like).")

    picked = with_depth if len(with_depth) <= args.scenes else rng.sample(with_depth, args.scenes)

    rows: list[dict] = []
    skipped = {"no_depth_frame": 0, "missing_file": 0, "load_err": 0}
    for sc in picked:
        frames = [fr for fr in sc["frames"] if fr.get("depth_path")]
        if not frames:
            skipped["no_depth_frame"] += 1
            continue
        sample = frames if len(frames) <= args.frames_per_scene else rng.sample(frames, args.frames_per_scene)
        for fr in sample:
            dpath = args.root / fr["depth_path"]
            ipath = args.root / fr["instance_mask_path"]
            if not dpath.exists() or not ipath.exists():
                skipped["missing_file"] += 1
                continue
            try:
                depth = load_depth(dpath)
                inst = load_inst(ipath)
                row = analyse_frame(depth, inst)
            except Exception as e:  # noqa: BLE001
                skipped["load_err"] += 1
                print(f"  load fail scene={sc.get('scene_id')} {dpath.name}: {e}")
                continue
            if row is None:
                continue
            row["scene_id"] = sc.get("scene_id")
            row["frame"] = Path(fr["depth_path"]).name
            rows.append(row)

    if not rows:
        raise SystemExit(f"No frames analysed. skipped={skipped}")

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def p90(key: str) -> float:
        return float(np.percentile([r[key] for r in rows], 90))

    n = len(rows)
    n_any_pollute = sum(1 for r in rows if r["polluted_frac_of_image"] > 0)
    n_heavy = sum(1 for r in rows if r.get("inst_area_lost_frac", 0) >= 0.05)
    n_id_half = sum(1 for r in rows if r["max_id_lost_frac"] >= 0.5)

    print(f"\nanalysed_frames={n} skipped={skipped}")
    print("--- aggregate (same valid_mask formula as dataset_manifest) ---")
    print(f"mean depth_invalid_frac          = {mean('invalid_frac'):.4f}  (p90 {p90('invalid_frac'):.4f})")
    print(f"mean polluted/image              = {mean('polluted_frac_of_image'):.4f}  "
          f"(invalid & inst!=0 / HxW)")
    print(f"mean polluted/invalid            = {mean('polluted_frac_of_invalid'):.4f}  "
          f"(of depth holes, fraction that had a real instance id)")
    print(f"mean fg instance area lost frac  = {mean('inst_area_lost_frac'):.4f}  "
          f"(p90 {p90('inst_area_lost_frac'):.4f})")
    print(f"frames with any polluted pixels  = {n_any_pollute}/{n} ({100*n_any_pollute/n:.1f}%)")
    print(f"frames losing >=5% fg inst area  = {n_heavy}/{n} ({100*n_heavy/n:.1f}%)")
    print(f"frames with some id losing >=50% = {n_id_half}/{n} ({100*n_id_half/n:.1f}%)")

    # Top offenders
    top = sorted(rows, key=lambda r: r.get("inst_area_lost_frac", 0), reverse=True)[:8]
    print("\n--- worst frames by fg area lost ---")
    for r in top:
        print(
            f"  {r['scene_id']} {r['frame']}: "
            f"invalid={r['invalid_frac']:.3f} "
            f"polluted/img={r['polluted_frac_of_image']:.4f} "
            f"fg_lost={r['inst_area_lost_frac']:.3f} "
            f"ids_touched={r['n_ids_touched']} "
            f"max_id_lost={r['max_id_lost_frac']:.2f}"
        )

    print("\n--- verdict ---")
    if n_any_pollute == 0:
        print(
            "NOT OBSERVED on this sample: depth holes never overlap non-zero instance ids.\n"
            "Code path in wrapper still exists, but data may not trigger I1 pollution."
        )
    else:
        print(
            "CONFIRMED on data: depth-invalid pixels carry non-zero instance ids.\n"
            "make_manifest only requires depth *file* presence; it does NOT filter holes.\n"
            "dataset_manifest correctly marks holes in valid_mask; the bug is wrapper\n"
            "reusing that mask as instance_valid_mask (semantic mix-up), not missing\n"
            "make_manifest gating."
        )


if __name__ == "__main__":
    main()

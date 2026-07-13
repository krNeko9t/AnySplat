#!/usr/bin/env python3
"""
Extract InsScene-15K processed_infinigen zips and build a single manifest.jsonl
for instseg training. Each zip is one subscene → one line in manifest.

Usage:
  # 1) Extract all zips and build manifest (need enough disk for extracted data)
  python scripts/extract_and_make_manifest_insscene.py \\
    --data_dir /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/dataset/InsScene-15K/processed_infinigen \\
    --extract_dir /path/to/processed_infinigen_extracted \\
    --manifest_out /path/to/processed_infinigen_extracted/manifest.jsonl

  # 2) Only build manifest (extract_dir already exists)
  python scripts/extract_and_make_manifest_insscene.py \\
    --data_dir ... --extract_dir ... --manifest_out ... --manifest_only

  # 3) Limit scenes for quick test
  python scripts/extract_and_make_manifest_insscene.py ... --max_subscenes 10

Then train with root=extract_dir, manifest_path=manifest_out, multi-GPU e.g.:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -m \\
    experiment=instseg_anysplat \\
    dataset.manifest.root=/path/to/processed_infinigen_extracted \\
    dataset.manifest.manifest_path=/path/to/processed_infinigen_extracted/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from make_manifest_infinigen import make_scene_manifest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def collect_zip_tasks(data_dir: Path, extract_dir: Path, max_subscenes: int | None):
    """Yield (zip_path, extract_to_dir, scene_id_prefix) for each zip."""
    data_dir = Path(data_dir)
    extract_dir = Path(extract_dir)
    n = 0
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for zip_path in sorted(scene_dir.glob("*.zip")):
            stem = zip_path.stem
            out_subdir = extract_dir / scene_dir.name / stem
            yield zip_path, out_subdir, f"{scene_dir.name}/{stem}"
            n += 1
            if max_subscenes is not None and n >= max_subscenes:
                return


def main():
    ap = argparse.ArgumentParser(description="Extract InsScene-15K zips and build manifest.jsonl for instseg.")
    ap.add_argument("--data_dir", type=str, required=True, help="processed_infinigen dir (scene_*/xxx.zip)")
    ap.add_argument("--extract_dir", type=str, required=True, help="Where to extract (e.g. fast SSD). Will create scene_XXX/subscene_name/")
    ap.add_argument("--manifest_out", type=str, required=True, help="Output manifest path (.jsonl)")
    ap.add_argument("--manifest_only", action="store_true", help="Skip extract; only build manifest from existing extract_dir")
    ap.add_argument("--max_subscenes", type=int, default=None, help="Max number of subscenes to process (for testing)")
    ap.add_argument("--extract_workers", type=int, default=4, help="Parallel workers for extraction")
    ap.add_argument("--camera", type=str, default="camera_0")
    ap.add_argument("--use_depth", choices=["npy", "png"], default="npy")
    ap.add_argument("--use_seg", choices=["npy", "png"], default="npy")
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    extract_dir = Path(args.extract_dir)
    manifest_path = Path(args.manifest_out)

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    tasks = list(collect_zip_tasks(data_dir, extract_dir, args.max_subscenes))
    if not tasks:
        raise ValueError(f"No scene_*/*.zip found under {data_dir}")

    # Extract (unless manifest_only)
    if not args.manifest_only:
        extract_dir.mkdir(parents=True, exist_ok=True)
        workers = min(args.extract_workers, len(tasks))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(extract_zip, zip_path, out_dir): (zip_path, out_dir) for zip_path, out_dir, _ in tasks}
            for fut in as_completed(futures):
                zip_path, out_dir = futures[fut]
                try:
                    fut.result()
                    done += 1
                    if done % 50 == 0 or done == len(tasks):
                        print(f"Extracted {done}/{len(tasks)}: {out_dir.relative_to(extract_dir)}")
                except Exception as e:
                    print(f"Failed {zip_path}: {e}")
        print(f"Extraction done: {done} subscenes under {extract_dir}")

    # Build manifest: one line per subscene, paths relative to extract_dir
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = []
    with manifest_path.open("w") as f:
        for zip_path, out_subdir, scene_id_prefix in tasks:
            if not out_subdir.exists():
                failed.append(str(out_subdir))
                continue
            try:
                scene_obj = make_scene_manifest(
                    out_subdir,
                    camera=args.camera,
                    use_depth=args.use_depth,
                    use_seg=args.use_seg,
                    scene_id=scene_id_prefix,
                    near=args.near,
                    far=args.far,
                )
            except Exception as e:
                failed.append(f"{out_subdir}: {e}")
                continue
            if len(scene_obj["frames"]) < 2:
                failed.append(f"{scene_id_prefix}: only {len(scene_obj['frames'])} frame(s), need >=2")
                continue
            # Paths in scene_obj are relative to out_subdir; we need relative to extract_dir
            prefix = f"{scene_id_prefix}/"
            for fr in scene_obj["frames"]:
                for key in ("rgb_path", "depth_path", "instance_mask_path"):
                    fr[key] = prefix + fr[key]
            f.write(json.dumps(scene_obj) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"Manifest: wrote {written} scenes")

    if failed:
        print(f"Skipped/failed {len(failed)} subscenes (first 5): {failed[:5]}")
    print(f"Wrote manifest: {manifest_path} ({written} scenes)")
    print("Train with:")
    print(f"  dataset.manifest.root={extract_dir.resolve()}")
    print(f"  dataset.manifest.manifest_path={manifest_path.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T8: AnySplat zero-shot reconstruction baseline on InsScene-15K val scenes.

Measures PSNR / SSIM / LPIPS of `hf:lhjiang/anysplat` on the fixed val split,
plus two probes asked for by the InstanceSplat map:
  - `voxelize_ratio` per (scene, view count)
  - peak CUDA memory per view count

Protocol (locked; see .scratch/instancesplat/tickets/T8-AnySplat零样本基线.md):
  - val split: manifest_val_instancesplat.jsonl (50 scenes: 25 spp + 25 re10k)
  - pure *reconstruction*: re-render the N input (context) views at the
    model's predicted poses (pose-free), exactly like AnySplatWrapper's
    validation_step (num_target_views = 0)
  - view counts N in {2, 4, 8}; views drawn by ViewSamplerBoundedFixed
    (stage="val", repo-default gap params), seeded per (scene, N)
  - resolution: long side 448, height snapped to the training aspect bins
    {0.5, 0.625, 0.75, 0.875, 1.0} x 448 (ties round up)
  - instance head disabled (instance_feat_dim = 0)

Usage (cluster, conda env anysplat):
  python scripts/zeroshot_baseline_anysplat.py \
      --root /mnt/storage_pool/liaoyuanjun/data/InsScene-15K \
      --manifest /mnt/storage_pool/liaoyuanjun/data/InsScene-15K/manifest_val_instancesplat.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_typed_config
from src.dataset.collate import collate_examples
from src.dataset.dataset_manifest import DatasetManifest, DatasetManifestCfg
from src.dataset.view_sampler.view_sampler_bounded_fixed import (
    ViewSamplerBoundedFixed,
    ViewSamplerBoundedFixedCfg,
)
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.model.arch import get_model
from src.model.arch.anysplat import EncoderAnySplatCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

ASPECT_BINS = (0.5, 0.625, 0.75, 0.875, 1.0)
LONG_SIDE = 448


def snap_aspect_bin(aspect: float) -> float:
    """Snap H/W to the nearest training aspect bin; ties round up."""
    return min(ASPECT_BINS, key=lambda b: (abs(b - aspect), -b))


def build_model(voxel_size: float) -> torch.nn.Module:
    enc = OmegaConf.load(CONFIG_DIR / "model/encoder/anysplat.yaml")
    enc.pretrained_weights = "hf:lhjiang/anysplat"
    enc.instance_feat_dim = 0  # pure reconstruction, no instance head
    # Match the released HF config (lhjiang/anysplat config.json):
    # pred_head_type=depth (a "point" head has no released weights) and
    # anchor_feat_dim=128.
    enc.pred_head_type = "depth"
    enc.anchor_feat_dim = 128
    enc.voxel_size = voxel_size
    enc.distill = False
    # voxelize defaults to False and is only set by config/experiment/*.yaml,
    # which this script does not load. The released HF config.json says
    # "voxelize": true, so the model was TRAINED with voxelization -- leaving it
    # off makes measurements wrong twice over:
    #   * infos["voxelize_ratio"] degenerates to (h*w*v)/(h*w*v) == 1.0 exactly
    #     (no voxels merged AND conf_valid_mask is all-True since
    #     render_conf is also False), so it is an identity, not a measurement;
    #   * PSNR/SSIM/LPIPS are then produced by a config the checkpoint never
    #     saw at training time.
    enc.voxelize = True
    encoder_cfg = load_typed_config(enc, EncoderAnySplatCfg)

    dec = OmegaConf.load(CONFIG_DIR / "model/decoder/splatting_cuda.yaml")
    decoder_cfg = load_typed_config(dec, DecoderSplattingCUDACfg)

    model = get_model(encoder_cfg, decoder_cfg)
    model.eval()
    return model


def build_dataset(root: Path, manifest: Path, max_views: int) -> DatasetManifest:
    vs_cfg = ViewSamplerBoundedFixedCfg(
        name="bounded_fixed",
        num_context_views=max_views,  # enables gap mapping for 2..max_views
        num_target_views=0,           # reconstruction only, no NVS target
        min_distance_between_context_views=12,
        max_distance_between_context_views=24,
        min_distance_to_context_views=0,
        warm_up_steps=0,
        initial_min_distance_between_context_views=2,
        initial_max_distance_between_context_views=6,
        max_img_per_gpu=24,
        min_gap_multiplier=3,
        max_gap_multiplier=5,
    )
    sampler = ViewSamplerBoundedFixed(
        vs_cfg, stage="val", is_overfitting=False,
        cameras_are_circular=False, step_tracker=None,
    )
    ds_cfg = DatasetManifestCfg(
        name="manifest",
        root=root,
        manifest_path=manifest,
        original_image_shape=[288, 512],
        input_image_shape=[224, 448],  # patchsize_w = 448 // 14 = 32
        background_color=[1.0, 1.0, 1.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=vs_cfg,
    )
    return DatasetManifest(ds_cfg, stage="val", view_sampler=sampler)


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path,
                        default=Path(".scratch/instancesplat/t8_zeroshot"))
    parser.add_argument("--num_views", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fixed_patch_h", type=int, default=None,
                        help="override aspect-bin patch height (e.g. 16 for 224x448, "
                             "the zero-shot model's training resolution)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first K scenes (smoke test)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[T8] loading model (hf:lhjiang/anysplat, voxel_size={args.voxel_size})...")
    model = build_model(args.voxel_size).to(device)
    data_shim = model.encoder.get_data_shim()

    dataset = build_dataset(args.root, args.manifest, max(args.num_views))
    n_scenes = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"[T8] val manifest: {args.manifest} ({len(dataset)} scenes, using {n_scenes})")

    results: list[dict] = []
    for num_views in args.num_views:
        torch.cuda.reset_peak_memory_stats()
        for i in range(n_scenes):
            scene = dataset.scenes[i]
            scene_id = str(scene.get("scene_id", i))
            subset = scene_id.split("_", 1)[0]

            # Deterministic per-(scene, N) view draw.
            torch.manual_seed(args.seed + i * 100 + num_views)

            hw = scene["frames"][0].get("HW")
            if hw is None:
                hw = dataset.cfg.original_image_shape
            aspect = min(float(hw[0]) / float(hw[1]), 1.0)
            bin_ = snap_aspect_bin(aspect)
            patch_h = int(round(LONG_SIDE * bin_)) // 14
            if args.fixed_patch_h is not None:
                patch_h = args.fixed_patch_h

            t0 = time.monotonic()
            example = dataset[i, num_views, patch_h]
            t_io = time.monotonic() - t0

            batch = collate_examples([example])
            batch = data_shim(batch)
            batch = to_device(batch, device)
            context_image = batch["context"]["image"]  # [-1, 1]
            model_input = (context_image + 1) / 2
            rgb_gt = model_input[0].float()  # [v, 3, h, w]

            t0 = time.monotonic()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoder_output, output = model(model_input, 0)
            torch.cuda.synchronize()
            t_fwd = time.monotonic() - t0

            rgb_pred = output.color[0].float()
            psnr = compute_psnr(rgb_gt, rgb_pred).mean().item()
            ssim = compute_ssim(rgb_gt, rgb_pred).mean().item()
            lpips = compute_lpips(rgb_gt, rgb_pred).mean().item()
            voxelize_ratio = float(encoder_output.infos["voxelize_ratio"])
            # Mean |pts_all| in the model's own canonical scale. Needed to judge
            # whether voxel_size=0.002 is a genuine no-op: compare it against
            # the per-pixel world-space footprint (scene_scale / focal_px).
            scene_scale = float(encoder_output.infos["scene_scale"])
            v, _, h, w = rgb_gt.shape
            gs_num = voxelize_ratio * h * w * v

            record = {
                "scene_id": scene_id,
                "subset": subset,
                "num_views": num_views,
                "aspect_bin": bin_,
                "patch_h": patch_h * 14,
                "context_indices": example["context"]["index"].tolist(),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "voxelize_ratio": voxelize_ratio,
                "gs_num": gs_num,
                "scene_scale": scene_scale,
                "t_io_s": round(t_io, 3),
                "t_fwd_s": round(t_fwd, 3),
            }
            results.append(record)
            print(
                f"[T8] N={num_views} [{i + 1}/{n_scenes}] {scene_id:<24} "
                f"bin={bin_:.3f} PSNR={psnr:.2f} SSIM={ssim:.4f} "
                f"LPIPS={lpips:.4f} vox_ratio={voxelize_ratio:.3f} "
                f"GS={gs_num / 1e6:.2f}M scale={scene_scale:.3f} fwd={t_fwd:.1f}s"
            )
            # Checkpoint after every scene so a crash loses nothing.
            (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))

        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        print(
            f"[T8] N={num_views} peak GPU memory: "
            f"allocated={peak_alloc:.2f} GiB, reserved={peak_reserved:.2f} GiB"
        )
        for r in results:
            if r["num_views"] == num_views:
                r["peak_mem_alloc_gib"] = peak_alloc
                r["peak_mem_reserved_gib"] = peak_reserved
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def agg(records, key):
        return sum(r[key] for r in records) / max(len(records), 1)

    lines = [
        "# T8 AnySplat 零样本基线（纯重建）",
        "",
        f"- manifest: `{args.manifest}`",
        f"- model: hf:lhjiang/anysplat (instance_feat_dim=0, voxel_size={args.voxel_size})",
        f"- seed: {args.seed}, scenes: {n_scenes}",
        "",
        "| 视角数 | 子集 | 场景数 | PSNR↑ | SSIM↑ | LPIPS↓ | voxelize_ratio | 峰值显存(GiB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for num_views in args.num_views:
        for subset in ("all", "spp", "re10k"):
            recs = [
                r for r in results
                if r["num_views"] == num_views
                and (subset == "all" or r["subset"] == subset)
            ]
            if not recs:
                continue
            lines.append(
                f"| {num_views} | {subset} | {len(recs)} "
                f"| {agg(recs, 'psnr'):.2f} | {agg(recs, 'ssim'):.4f} "
                f"| {agg(recs, 'lpips'):.4f} | {agg(recs, 'voxelize_ratio'):.3f} "
                f"| {recs[0]['peak_mem_alloc_gib']:.2f} |"
            )
    summary = "\n".join(lines)
    (args.out_dir / "summary.md").write_text(summary + "\n")
    print("\n" + summary + "\n")
    print(f"[T8] results -> {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

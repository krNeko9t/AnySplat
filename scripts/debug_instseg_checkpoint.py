# deprecated: use debug_instseg_ckpt.py instead
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# NOTE: Keep optional deps (omegaconf, PIL) as lazy imports inside _main()
# so `python ... -h` works even in minimal envs.


@dataclass
class FixedViews:
    scene_id: str
    context: list[int]
    target: list[int]


def _to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    """img_chw: [3,H,W] float in [0,1]"""
    img = img_chw.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return img


def _save_image(path: Path, arr_uint8_hwc: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8_hwc).save(path)


def _make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    """Return [num_colors,3] uint8 LUT."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    # avoid too-dark colors
    lut = np.maximum(lut, 32).astype(np.uint8)
    return lut


def _colorize_labels(labels_hw: np.ndarray, lut: np.ndarray, ignore_label: int = 0) -> np.ndarray:
    """labels_hw: [H,W] int. lut: [K,3]. ignore -> black."""
    h, w = labels_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    mask = labels_hw != ignore_label
    if mask.any():
        idx = labels_hw[mask]
        idx = np.clip(idx, 0, lut.shape[0] - 1)
        out[mask] = lut[idx]
    return out


def _stack_h(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _unproject_depth_to_world(
    depth_vhw: torch.Tensor,
    intr_v33_norm: torch.Tensor,
    c2w_v44: torch.Tensor,
) -> torch.Tensor:
    """Return world points [V,H,W,3] using normalized intrinsics.

    intrinsics are normalized by image size: fx'=fx/W, fy'=fy/H, cx'=cx/W, cy'=cy/H.
    """
    assert depth_vhw.ndim == 3
    V, H, W = depth_vhw.shape
    device = depth_vhw.device
    dtype = depth_vhw.dtype

    fx = intr_v33_norm[:, 0, 0] * float(W)
    fy = intr_v33_norm[:, 1, 1] * float(H)
    cx = intr_v33_norm[:, 0, 2] * float(W)
    cy = intr_v33_norm[:, 1, 2] * float(H)

    u = torch.arange(W, device=device, dtype=dtype)[None, None, :].expand(V, H, W)
    v = torch.arange(H, device=device, dtype=dtype)[None, :, None].expand(V, H, W)

    z = depth_vhw
    x = (u - cx[:, None, None]) * z / (fx[:, None, None] + 1e-8)
    y = (v - cy[:, None, None]) * z / (fy[:, None, None] + 1e-8)

    cam = torch.stack([x, y, z], dim=-1)  # [V,H,W,3]
    R = c2w_v44[:, :3, :3]  # [V,3,3]
    t = c2w_v44[:, :3, 3]  # [V,3]
    world = (cam @ R.transpose(-1, -2)[:, None, None]) + t[:, None, None, :]
    return world


@torch.no_grad()
def _kmeans_global_cluster(
    feat_vnhw: torch.Tensor,
    valid_vhw: torch.Tensor,
    k: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster all views jointly with k-means on a subsample; assign all pixels by nearest center.

    Returns:
      pred_labels_vhw: int32 [V,H,W]
      centers: float32 [k,N]
    """
    from src.instseg.kmeans import kmeans_torch

    V, N, H, W = feat_vnhw.shape
    feat = feat_vnhw.permute(0, 2, 3, 1).reshape(-1, N)  # [P,N]
    valid = valid_vhw.reshape(-1)
    idx_all = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        pred = torch.zeros((V * H * W,), dtype=torch.int64, device=feat.device)
        return pred.view(V, H, W).cpu().numpy().astype(np.int32), np.zeros((k, N), np.float32)

    S = min(int(max_points), int(idx_all.numel()))
    g = torch.Generator(device=feat.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(idx_all.numel(), generator=g, device=feat.device)[:S]
    idx = idx_all[perm]

    x = feat[idx]
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    labels_s, centers = kmeans_torch(x, k=k, num_iters=30, seed=seed)

    # Assign all valid pixels to nearest center (cosine distance since normalized).
    feat_n = F.normalize(feat, p=2, dim=-1, eps=1e-8)
    dot = feat_n @ centers.T  # [P,k]
    pred = (2.0 - 2.0 * dot).argmin(dim=1).to(torch.int64)  # [P]
    pred[~valid] = 0

    return pred.view(V, H, W).cpu().numpy().astype(np.int32), centers.detach().cpu().numpy().astype(np.float32)


def _load_fixed_views(path: Path) -> FixedViews:
    obj = json.loads(path.read_text())
    return FixedViews(
        scene_id=str(obj["scene_id"]),
        context=[int(x) for x in obj["context"]],
        target=[int(x) for x in obj["target"]],
    )


def _find_scene_index(scenes: list[dict[str, Any]], scene_id: str) -> int:
    for i, s in enumerate(scenes):
        if str(s.get("scene_id", i)) == str(scene_id):
            return i
    raise ValueError(f"scene_id={scene_id} not found in manifest ({len(scenes)} scenes)")


def _load_lightning_ckpt(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # fallback: treat as raw state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Hydra run dir containing .hydra/config.yaml and fixed_views.json")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to Lightning .ckpt (default: run_dir/checkpoints/last.ckpt)")
    ap.add_argument("--out_dir", type=str, default="outputs/instseg_debug")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--k", type=int, default=20, help="k-means clusters")
    ap.add_argument("--max_points", type=int, default=200000, help="max sampled pixels for k-means init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / ".hydra" / "config.yaml"
    fixed_path = run_dir / "fixed_views.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not fixed_path.exists():
        raise FileNotFoundError(fixed_path)

    # Load Hydra config and build typed config using the repo's loader.
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.load(str(cfg_path))
    with torch.no_grad():
        # Import inside main to reuse repo path assumptions.
        from src.config import load_typed_root_config
        from src.dataset.dataset_custom import DatasetCustom
        from src.global_cfg import set_cfg
        from src.loss import get_losses
        from src.misc.step_tracker import StepTracker
        from src.model.model import get_model
        from src.model.model_wrapper import ModelWrapper

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Build model wrapper and load checkpoint weights.
    step_tracker = StepTracker()
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    wrapper = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), step_tracker)
    state_dict = _load_lightning_ckpt(ckpt_path)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[debug] Loaded ckpt={ckpt_path}")
    print(f"[debug] missing={len(missing)} unexpected={len(unexpected)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    wrapper.eval()

    fixed = _load_fixed_views(fixed_path)
    print(f"[debug] fixed_views scene_id={fixed.scene_id} context={fixed.context} target={fixed.target}")

    # Get custom dataset cfg from cfg.dataset wrappers.
    custom_cfg = None
    for w in cfg.dataset:
        if hasattr(w, "custom"):
            custom_cfg = w.custom
            break
    if custom_cfg is None:
        raise ValueError("No custom dataset cfg found in cfg.dataset")

    # Build dataset; we will override the sampler with fixed indices.
    class _FixedSampler:
        def __init__(self, ctx: list[int], tgt: list[int]):
            self.ctx = torch.tensor(ctx, dtype=torch.int64)
            self.tgt = torch.tensor(tgt, dtype=torch.int64)

        def sample(self, scene: str, num_context_views: int, extrinsics, intrinsics, device=torch.device("cpu")):
            return self.ctx.to(device), self.tgt.to(device), torch.tensor([0.5], device=device, dtype=torch.float32)

        @property
        def num_context_views(self) -> int:
            return int(self.ctx.numel())

        @property
        def num_target_views(self) -> int:
            return int(self.tgt.numel())

    ds = DatasetCustom(custom_cfg, "train", _FixedSampler([0, 1], [2]))
    scene_index = _find_scene_index(ds.scenes, fixed.scene_id)
    ds.view_sampler = _FixedSampler(fixed.context, fixed.target)  # type: ignore[assignment]

    ps_h = int(custom_cfg.input_image_shape[0] // 14)
    ps_w = int(custom_cfg.input_image_shape[1] // 14)
    example = ds.getitem(scene_index, len(fixed.context), (ps_h, ps_w))

    # Prepare inputs.
    ctx_img = example["context"]["image"].unsqueeze(0).to(device)  # [1,Vc,3,H,W] in [0,1]
    tgt_img = example["target"]["image"].unsqueeze(0).to(device)   # [1,Vt,3,H,W]
    images = torch.cat([ctx_img, tgt_img], dim=1)

    ctx_inst = example["context"]["instance_mask"].unsqueeze(0).to(device)  # [1,Vc,H,W]
    tgt_inst = example["target"]["instance_mask"].unsqueeze(0).to(device)   # [1,Vt,H,W]
    gt_inst = torch.cat([ctx_inst, tgt_inst], dim=1)  # [1,V,H,W]

    ctx_valid = example["context"]["valid_mask"].unsqueeze(0).to(device)
    tgt_valid = example["target"]["valid_mask"].unsqueeze(0).to(device)
    valid = torch.cat([ctx_valid, tgt_valid], dim=1).to(torch.bool)  # [1,V,H,W]

    # Optional: world points (for future KNN smoothing).
    # Use GT depth + GT cameras; intrinsics are normalized by image size.
    ctx_depth = example["context"]["depth"].unsqueeze(0).to(device)  # [1,Vc,H,W]
    tgt_depth = example["target"]["depth"].unsqueeze(0).to(device)
    depth = torch.cat([ctx_depth, tgt_depth], dim=1)  # [1,V,H,W]
    ctx_K = example["context"]["intrinsics"].unsqueeze(0).to(device)  # [1,Vc,3,3]
    tgt_K = example["target"]["intrinsics"].unsqueeze(0).to(device)
    K = torch.cat([ctx_K, tgt_K], dim=1)  # [1,V,3,3]
    ctx_c2w = example["context"]["extrinsics"].unsqueeze(0).to(device)  # [1,Vc,4,4]
    tgt_c2w = example["target"]["extrinsics"].unsqueeze(0).to(device)
    c2w = torch.cat([ctx_c2w, tgt_c2w], dim=1)  # [1,V,4,4]

    world_points = _unproject_depth_to_world(depth[0], K[0], c2w[0])

    # Forward encoder to get instance features.
    with torch.no_grad():
        enc_out = wrapper.model.encoder(images, global_step=0, visualization_dump=None)
    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        raise ValueError("instance_feat_map is None. Did you train with model.encoder.instance_feat_dim > 0?")

    feat_map = feat_map[0].float()  # [V,N,H,W]
    V, N, H, W = feat_map.shape
    valid_vhw = valid[0]

    # Global clustering across all views.
    pred_labels_vhw, centers = _kmeans_global_cluster(
        feat_map, valid_vhw, k=args.k, max_points=args.max_points, seed=args.seed
    )

    # Build GT label remapping to contiguous for colormap (keep 0 as ignore).
    gt_np = gt_inst[0].detach().cpu().numpy().astype(np.int32)  # [V,H,W]
    unique_ids = np.unique(gt_np)
    unique_ids = unique_ids[unique_ids != 0]
    id_to_contig = {int(i): (j + 1) for j, i in enumerate(unique_ids.tolist())}
    gt_contig = np.zeros_like(gt_np, dtype=np.int32)
    for i, j in id_to_contig.items():
        gt_contig[gt_np == i] = int(j)

    # Color LUTs (pred and GT use different LUT sizes).
    pred_lut = _make_color_lut(max(1, int(args.k) + 1), seed=0)
    gt_lut = _make_color_lut(max(1, int(len(unique_ids) + 1)), seed=1)

    # Save per-view outputs.
    view_indices = example["context"]["index"].tolist() + example["target"]["index"].tolist()
    (out_dir / "pred_labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_labels").mkdir(parents=True, exist_ok=True)
    for vi in range(V):
        frame_idx = int(view_indices[vi]) if vi < len(view_indices) else vi
        rgb = _to_uint8_rgb(images[0, vi])
        pred_col = _colorize_labels(pred_labels_vhw[vi], pred_lut, ignore_label=0)
        gt_col = _colorize_labels(gt_contig[vi], gt_lut, ignore_label=0)

        _save_image(out_dir / "rgb" / f"{frame_idx:04d}.png", rgb)
        _save_image(out_dir / "pred" / f"{frame_idx:04d}.png", pred_col)
        _save_image(out_dir / "gt" / f"{frame_idx:04d}.png", gt_col)

        trip = _stack_h([rgb, pred_col, gt_col])
        _save_image(out_dir / "triplet" / f"{frame_idx:04d}.png", trip)

        np.save(out_dir / "pred_labels" / f"{frame_idx:04d}.npy", pred_labels_vhw[vi].astype(np.int32))
        np.save(out_dir / "gt_labels" / f"{frame_idx:04d}.npy", gt_contig[vi].astype(np.int32))

    # Save metadata for reproducibility.
    meta = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "scene_id": fixed.scene_id,
        "context": fixed.context,
        "target": fixed.target,
        "k": int(args.k),
        "max_points": int(args.max_points),
        "seed": int(args.seed),
        "H": int(H),
        "W": int(W),
        "N": int(N),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[debug] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    _main()


"""
deprecated: use instseg_infer.py instead
Debug a trained instseg checkpoint: run encoder + k-means on one scene and save visualizations.

Usage:
  # Random views (no fixed_views.json needed):
  python scripts/debug_instseg_checkpoint.py --run_dir output/.../run --ckpt path/to.ckpt

  # Specify scene by id or index:
  python ... --scene my_scene_001
  python ... --scene 0

  # Specify which views to use (context + target):
  python ... --context 0,1,5 --target 2,3
  python ... --fixed_views /path/to/views.json   # JSON: {"scene_id": "...", "context": [0,1], "target": [2]}

  # When using random views, control how many:
  python ... --num_context 4 --num_target 2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return img


def _save_image(path: Path, arr_uint8_hwc: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8_hwc).save(path)


def _make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    lut = np.maximum(lut, 32).astype(np.uint8)
    return lut


def _colorize_labels(labels_hw: np.ndarray, lut: np.ndarray, ignore_label: int = 0) -> np.ndarray:
    h, w = labels_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    mask = labels_hw != ignore_label
    if mask.any():
        idx = np.clip(labels_hw[mask], 0, lut.shape[0] - 1)
        out[mask] = lut[idx]
    return out


def _stack_h(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _unproject_depth_to_world(
    depth_vhw: torch.Tensor,
    intr_v33_norm: torch.Tensor,
    c2w_v44: torch.Tensor,
) -> torch.Tensor:
    V, H, W = depth_vhw.shape
    device, dtype = depth_vhw.device, depth_vhw.dtype
    fx = intr_v33_norm[:, 0, 0] * float(W)
    fy = intr_v33_norm[:, 1, 1] * float(H)
    cx = intr_v33_norm[:, 0, 2] * float(W)
    cy = intr_v33_norm[:, 1, 2] * float(H)
    u = torch.arange(W, device=device, dtype=dtype)[None, None, :].expand(V, H, W)
    v = torch.arange(H, device=device, dtype=dtype)[None, :, None].expand(V, H, W)
    z = depth_vhw
    x = (u - cx[:, None, None]) * z / (fx[:, None, None] + 1e-8)
    y = (v - cy[:, None, None]) * z / (fy[:, None, None] + 1e-8)
    cam = torch.stack([x, y, z], dim=-1)
    R = c2w_v44[:, :3, :3]
    t = c2w_v44[:, :3, 3]
    world = (cam @ R.transpose(-1, -2)[:, None, None]) + t[:, None, None, :]
    return world


@torch.no_grad()
def _kmeans_global_cluster(
    feat_vnhw: torch.Tensor,
    valid_vhw: torch.Tensor,
    k: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from src.instseg.kmeans import kmeans_torch
    V, N, H, W = feat_vnhw.shape
    feat = feat_vnhw.permute(0, 2, 3, 1).reshape(-1, N)
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
    _, centers = kmeans_torch(x, k=k, num_iters=30, seed=seed)
    feat_n = F.normalize(feat, p=2, dim=-1, eps=1e-8)
    dot = feat_n @ centers.T
    pred = (2.0 - 2.0 * dot).argmin(dim=1).to(torch.int64)
    pred[~valid] = 0
    return pred.view(V, H, W).cpu().numpy().astype(np.int32), centers.detach().cpu().numpy().astype(np.float32)


def _load_lightning_ckpt(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def _find_scene_index(scenes: list[dict[str, Any]], scene_id: str) -> int:
    for i, s in enumerate(scenes):
        if str(s.get("scene_id", i)) == str(scene_id):
            return i
    raise ValueError(f"scene_id={scene_id} not found in manifest ({len(scenes)} scenes)")


def _parse_views_from_args(args: argparse.Namespace) -> tuple[str | int, list[int] | None, list[int] | None]:
    """Returns (scene_id_or_index, context_indices or None, target_indices or None). None = use random."""
    scene: str | int
    context: list[int] | None = None
    target: list[int] | None = None
    if getattr(args, "fixed_views", None):
        path = Path(args.fixed_views)
        if not path.exists():
            raise FileNotFoundError(path)
        obj = json.loads(path.read_text())
        scene = str(obj["scene_id"])
        context = [int(x) for x in obj["context"]]
        target = [int(x) for x in obj["target"]]
        return scene, context, target
    if getattr(args, "scene", None) is not None:
        s = args.scene
        scene = int(s) if isinstance(s, str) and s.isdigit() else str(s)
    else:
        scene = 0
    if getattr(args, "context", None) and getattr(args, "target", None):
        context = [int(x.strip()) for x in args.context.split(",")]
        target = [int(x.strip()) for x in args.target.split(",")]
    return scene, context, target


def _make_debug_sampler(
    context: list[int] | None,
    target: list[int] | None,
    num_context: int,
    num_target: int,
    seed: int,
):
    if context is not None and target is not None:
        ctx_t = torch.tensor(context, dtype=torch.int64)
        tgt_t = torch.tensor(target, dtype=torch.int64)
        class Fixed:
            @property
            def num_context_views(self) -> int:
                return int(ctx_t.numel())
            @property
            def num_target_views(self) -> int:
                return int(tgt_t.numel())
            def sample(self, scene: str, num_context_views: int, extrinsics, intrinsics, device=torch.device("cpu")):
                return ctx_t.to(device), tgt_t.to(device), torch.tensor([0.5], device=device, dtype=torch.float32)
        return Fixed()
    num_c, num_t = num_context, num_target
    class Random:
        _seed = seed
        @property
        def num_context_views(self) -> int:
            return num_c
        @property
        def num_target_views(self) -> int:
            return num_t
        def sample(self, scene: str, num_context_views: int, extrinsics, intrinsics, device=torch.device("cpu")):
            n = extrinsics.shape[0]
            need = num_c + num_t
            if n < need:
                raise ValueError(f"Scene has {n} views but need at least {need} (num_context={num_c}, num_target={num_t})")
            rng = random.Random(self._seed)
            indices = list(range(n))
            rng.shuffle(indices)
            ctx = indices[:num_c]
            tgt = indices[num_c : num_c + num_t]
            return (
                torch.tensor(ctx, dtype=torch.int64, device=device),
                torch.tensor(tgt, dtype=torch.int64, device=device),
                torch.tensor([0.5], device=device, dtype=torch.float32),
            )
    return Random()


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Debug instseg checkpoint: one scene, encoder + k-means, save rgb/pred/gt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run_dir", type=str, required=True, help="Hydra run dir (must contain .hydra/config.yaml)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--out_dir", type=str, default="outputs/instseg_debug")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scene", type=str, default=None, help="Scene: scene_id (str) or index (int). Default: 0")
    ap.add_argument("--context", type=str, default=None, help="Context view indices, comma-separated e.g. 0,1,5")
    ap.add_argument("--target", type=str, default=None, help="Target view indices, comma-separated e.g. 2,3")
    ap.add_argument("--fixed_views", type=str, default=None, help="Path to JSON {scene_id, context: [], target: []}. Overrides --scene/--context/--target")
    ap.add_argument("--num_context", type=int, default=2, help="When using random views: number of context views")
    ap.add_argument("--num_target", type=int, default=1, help="When using random views: number of target views")
    ap.add_argument("--k", type=int, default=20, help="K-means clusters")
    ap.add_argument("--max_points", type=int, default=200000, help="Max pixels for k-means init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    scene_spec, context_list, target_list = _parse_views_from_args(args)
    use_random = context_list is None or target_list is None
    if use_random and (getattr(args, "context", None) or getattr(args, "target", None)):
        context_list = None
        target_list = None
    sampler = _make_debug_sampler(
        context_list, target_list,
        args.num_context, args.num_target,
        args.seed,
    )

    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.load(str(cfg_path))
    from src.config import load_typed_root_config
    from src.dataset.dataset_custom import DatasetCustom
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.model import get_model
    from src.model.model_wrapper import ModelWrapper

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    step_tracker = StepTracker()
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    wrapper = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), step_tracker)
    state_dict = _load_lightning_ckpt(ckpt_path)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[debug] Loaded ckpt={ckpt_path}, missing={len(missing)} unexpected={len(unexpected)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    wrapper.eval()

    custom_cfg = None
    for w in cfg.dataset:
        if hasattr(w, "custom"):
            custom_cfg = w.custom
            break
    if custom_cfg is None:
        raise ValueError("No custom dataset cfg in cfg.dataset")

    ds = DatasetCustom(custom_cfg, "train", sampler)
    if isinstance(scene_spec, int):
        scene_index = scene_spec
        if scene_index < 0 or scene_index >= len(ds.scenes):
            raise ValueError(f"Scene index {scene_index} out of range [0, {len(ds.scenes)})")
        scene_id = str(ds.scenes[scene_index].get("scene_id", scene_index))
    else:
        scene_index = _find_scene_index(ds.scenes, scene_spec)
        scene_id = scene_spec

    print(f"[debug] scene_id={scene_id} scene_index={scene_index} views={'random' if use_random else 'fixed'} context={context_list} target={target_list}")

    ps_h = int(custom_cfg.input_image_shape[0] // 14)
    ps_w = int(custom_cfg.input_image_shape[1] // 14)
    num_context_views = sampler.num_context_views
    with torch.no_grad():
        example = ds.getitem(scene_index, num_context_views, (ps_h, ps_w))

    ctx_img = example["context"]["image"].unsqueeze(0).to(device)
    tgt_img = example["target"]["image"].unsqueeze(0).to(device)
    images = torch.cat([ctx_img, tgt_img], dim=1)
    ctx_inst = example["context"]["instance_mask"].unsqueeze(0).to(device)
    tgt_inst = example["target"]["instance_mask"].unsqueeze(0).to(device)
    gt_inst = torch.cat([ctx_inst, tgt_inst], dim=1)
    ctx_valid = example["context"]["valid_mask"].unsqueeze(0).to(device)
    tgt_valid = example["target"]["valid_mask"].unsqueeze(0).to(device)
    valid = torch.cat([ctx_valid, tgt_valid], dim=1).to(torch.bool)
    ctx_depth = example["context"]["depth"].unsqueeze(0).to(device)
    tgt_depth = example["target"]["depth"].unsqueeze(0).to(device)
    depth = torch.cat([ctx_depth, tgt_depth], dim=1)
    ctx_K = example["context"]["intrinsics"].unsqueeze(0).to(device)
    tgt_K = example["target"]["intrinsics"].unsqueeze(0).to(device)
    K = torch.cat([ctx_K, tgt_K], dim=1)
    ctx_c2w = example["context"]["extrinsics"].unsqueeze(0).to(device)
    tgt_c2w = example["target"]["extrinsics"].unsqueeze(0).to(device)
    c2w = torch.cat([ctx_c2w, tgt_c2w], dim=1)

    with torch.no_grad():
        enc_out = wrapper.model.encoder(images, global_step=0, visualization_dump=None)
    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        raise ValueError("encoder did not return instance_feat_map (instance_feat_dim > 0?)")
    feat_map = feat_map[0].float()
    V, N, H, W = feat_map.shape
    valid_vhw = valid[0]
    pred_labels_vhw, centers = _kmeans_global_cluster(
        feat_map, valid_vhw, k=args.k, max_points=args.max_points, seed=args.seed
    )

    gt_np = gt_inst[0].detach().cpu().numpy().astype(np.int32)
    unique_ids = np.unique(gt_np)
    unique_ids = unique_ids[unique_ids != 0]
    id_to_contig = {int(i): (j + 1) for j, i in enumerate(unique_ids.tolist())}
    gt_contig = np.zeros_like(gt_np, dtype=np.int32)
    for i, j in id_to_contig.items():
        gt_contig[gt_np == i] = int(j)
    pred_lut = _make_color_lut(max(1, args.k + 1), seed=0)
    gt_lut = _make_color_lut(max(1, len(unique_ids) + 1), seed=1)

    view_indices = example["context"]["index"].tolist() + example["target"]["index"].tolist()
    actual_context = example["context"]["index"].tolist()
    actual_target = example["target"]["index"].tolist()

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

    meta = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "scene_id": scene_id,
        "scene_index": scene_index,
        "context": actual_context,
        "target": actual_target,
        "random_views": use_random,
        "k": args.k,
        "max_points": args.max_points,
        "seed": args.seed,
        "H": int(H),
        "W": int(W),
        "N": int(N),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[debug] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    _main()
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FixedViews:
    scene_id: str
    context: list[int]
    target: list[int]


def _make_color_lut(num_colors: int, seed: int = 0) -> np.ndarray:
    """Return [num_colors,3] uint8 LUT. idx=0 reserved for black."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    lut = np.maximum(lut, 32).astype(np.uint8)  # avoid too-dark colors
    lut[0] = 0
    return lut


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
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def _parse_indices_arg(s: str | None) -> list[int]:
    if not s:
        return []
    parts = [x.strip() for x in str(s).split(",")]
    return [int(x) for x in parts if x]


@torch.no_grad()
def _cluster_gaussians(
    feat_gn: torch.Tensor,
    opacities_g: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    max_points: int,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster gaussian instance embeddings with k-means (cosine).

    Returns:
      labels_g: int32 [G] with 0=ignore, 1..k=cluster id
      centers: float32 [k,N]
    """
    from src.instseg.kmeans import kmeans_torch

    G, N = feat_gn.shape
    if opacities_g.ndim != 1:
        opacities_g = opacities_g.reshape(-1)
    # opacities_g is already in probability space [0,1] (after map_pdf_to_opacity).
    # Do NOT apply sigmoid again.
    valid = opacities_g > float(opacity_threshold)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        labels = np.zeros((G,), np.int32)
        centers = np.zeros((k, N), np.float32)
        return labels, centers

    # Subsample for kmeans fit.
    S = min(int(max_points), int(valid_idx.numel()))
    g = torch.Generator(device=feat_gn.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(valid_idx.numel(), generator=g, device=feat_gn.device)[:S]
    sample_idx = valid_idx[perm]

    x = feat_gn[sample_idx].float()
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    _, centers = kmeans_torch(x, k=int(k), num_iters=int(iters), seed=int(seed))

    # Assign all valid gaussians to nearest center by cosine distance (since normalized).
    feat_n = F.normalize(feat_gn.float(), p=2, dim=-1, eps=1e-8)
    centers_n = F.normalize(centers.float(), p=2, dim=-1, eps=1e-8)
    dot = feat_n @ centers_n.T  # [G,k]
    pred0 = dot.argmax(dim=1).to(torch.int64)  # [G] in 0..k-1
    pred0[~valid] = -1

    labels = torch.zeros((G,), dtype=torch.int32, device=feat_gn.device)
    keep = pred0 >= 0
    labels[keep] = (pred0[keep] + 1).to(torch.int32)  # shift to 1..k
    return labels.cpu().numpy().astype(np.int32), centers.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _cluster_gaussians_dbscan(
    feat_gn: torch.Tensor,
    opacities_g: torch.Tensor,
    eps: float,
    min_samples: int,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster gaussian instance embeddings with DBSCAN (cosine).

    Returns:
      labels_g: int32 [G] with 0=ignore, 1..C=cluster id (C is data-dependent)
      centers: float32 [C,N] (mean feature per cluster, L2-normalized)
    """
    try:
        from sklearn.cluster import DBSCAN  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DBSCAN clustering requires scikit-learn. "
            "Please install it with `pip install scikit-learn`."
        ) from exc

    G, N = feat_gn.shape
    if opacities_g.ndim != 1:
        opacities_g = opacities_g.reshape(-1)

    # Filter by opacity threshold, same语义 as k-means版本.
    valid = opacities_g > float(opacity_threshold)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        labels = np.zeros((G,), np.int32)
        centers = np.zeros((0, N), np.float32)
        return labels, centers

    # Normalize features for cosine metric.
    x = feat_gn[valid_idx].float()
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    x_np = x.detach().cpu().numpy()

    db = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="cosine",
        n_jobs=-1,
    )
    labels_valid = db.fit_predict(x_np)  # [num_valid], -1=noise

    labels = np.zeros((G,), np.int32)
    valid_idx_np = valid_idx.detach().cpu().numpy()

    # Map DBSCAN labels: -1 -> 0(ignore), 0..C-1 -> 1..C.
    if labels_valid.size > 0:
        is_cluster = labels_valid >= 0
        if np.any(is_cluster):
            labels_shifted = np.zeros_like(labels_valid, dtype=np.int32)
            labels_shifted[is_cluster] = labels_valid[is_cluster] + 1
            labels[valid_idx_np] = labels_shifted.astype(np.int32)

    # Compute simple mean feature center per cluster (optional, for diagnostics).
    if np.any(labels_valid >= 0):
        num_clusters = int(labels_valid[labels_valid >= 0].max() + 1)
    else:
        num_clusters = 0
    centers = np.zeros((num_clusters, N), np.float32)
    if num_clusters > 0:
        for cid in range(num_clusters):
            mem_mask = labels_valid == cid
            if not np.any(mem_mask):
                continue
            mem_idx_np = valid_idx_np[mem_mask]
            mem_idx = torch.from_numpy(mem_idx_np).to(feat_gn.device)
            mem_feat = feat_gn[mem_idx].float()
            c = mem_feat.mean(dim=0)
            c = F.normalize(c, p=2, dim=-1, eps=1e-8)
            centers[cid] = c.detach().cpu().numpy().astype(np.float32)

    return labels.astype(np.int32), centers


@torch.no_grad()
def _cluster_gaussians_hdbscan(
    feat_gn: torch.Tensor,
    opacities_g: torch.Tensor,
    min_cluster_size: int,
    min_samples: int,
    opacity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster gaussian instance embeddings with HDBSCAN.

    Returns:
      labels_g: int32 [G] with 0=ignore, 1..C=cluster id
      centers: float32 [C,N] (mean feature per cluster, L2-normalized)
    """
    try:
        import hdbscan  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "HDBSCAN clustering requires the `hdbscan` package. "
            "Please install it with `pip install hdbscan`."
        ) from exc

    G, N = feat_gn.shape
    if opacities_g.ndim != 1:
        opacities_g = opacities_g.reshape(-1)

    valid = opacities_g > float(opacity_threshold)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        labels = np.zeros((G,), np.int32)
        centers = np.zeros((0, N), np.float32)
        return labels, centers

    x = feat_gn[valid_idx].float()
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    x_np = x.detach().cpu().numpy()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    labels_valid = clusterer.fit_predict(x_np)  # [num_valid], -1=noise

    labels = np.zeros((G,), np.int32)
    valid_idx_np = valid_idx.detach().cpu().numpy()

    if labels_valid.size > 0:
        is_cluster = labels_valid >= 0
        if np.any(is_cluster):
            labels_shifted = np.zeros_like(labels_valid, dtype=np.int32)
            labels_shifted[is_cluster] = labels_valid[is_cluster] + 1
            labels[valid_idx_np] = labels_shifted.astype(np.int32)

    if np.any(labels_valid >= 0):
        num_clusters = int(labels_valid[labels_valid >= 0].max() + 1)
    else:
        num_clusters = 0
    centers = np.zeros((num_clusters, N), np.float32)
    if num_clusters > 0:
        for cid in range(num_clusters):
            mem_mask = labels_valid == cid
            if not np.any(mem_mask):
                continue
            mem_idx_np = valid_idx_np[mem_mask]
            mem_idx = torch.from_numpy(mem_idx_np).to(feat_gn.device)
            mem_feat = feat_gn[mem_idx].float()
            c = mem_feat.mean(dim=0)
            c = F.normalize(c, p=2, dim=-1, eps=1e-8)
            centers[cid] = c.detach().cpu().numpy().astype(np.float32)

    return labels.astype(np.int32), centers


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Hydra run dir containing .hydra/config.yaml and fixed_views.json")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--out_dir", type=str, default="outputs/instseg_cluster_ply")
    ap.add_argument("--out_name", type=str, default="gaussians_cluster_color.ply")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--cluster_algo",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan", "hdbscan"],
        help="Which clustering algorithm to use for gaussian_instance_feat.",
    )
    ap.add_argument(
        "--scene_id",
        type=str,
        default="",
        help="Optional scene id to override fixed_views.json scene_id.",
    )
    ap.add_argument(
        "--context_views",
        type=str,
        default="",
        help="Optional comma-separated context view indices, e.g. '0,1,2'. If empty, views are sampled randomly.",
    )
    ap.add_argument(
        "--target_views",
        type=str,
        default="",
        help="Optional comma-separated target view indices, e.g. '3,4'. If empty, views are sampled randomly.",
    )
    ap.add_argument(
        "--num_context_views",
        type=int,
        default=2,
        help="Number of context views when sampling randomly (ignored if context_views provided).",
    )
    ap.add_argument(
        "--num_target_views",
        type=int,
        default=1,
        help="Number of target views when sampling randomly (ignored if target_views provided).",
    )
    ap.add_argument(
        "--use_fixed_views",
        action="store_true",
        help="If set, use scene_id/context/target from fixed_views.json instead of random sampling when CLI views are not provided.",
    )
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max_points", type=int, default=200000, help="Max gaussians used to fit kmeans")
    ap.add_argument("--dbscan_eps", type=float, default=0.3, help="DBSCAN eps (cosine distance).")
    ap.add_argument("--dbscan_min_samples", type=int, default=10, help="DBSCAN min_samples.")
    ap.add_argument("--hdbscan_min_cluster_size", type=int, default=50, help="HDBSCAN min_cluster_size.")
    ap.add_argument("--hdbscan_min_samples", type=int, default=10, help="HDBSCAN min_samples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--palette_seed", type=int, default=0)
    ap.add_argument("--opacity_threshold", type=float, default=1.0 / 255.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / ".hydra" / "config.yaml"
    fixed_path = run_dir / "fixed_views.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # Load Hydra config and build typed config using the repo's loader.
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.load(str(cfg_path))
    from src.config import load_typed_root_config
    from src.dataset.dataset_custom import DatasetCustom
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.model import get_model
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.post_opt.utils import rgb_to_sh

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    step_tracker = StepTracker()
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    wrapper = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), step_tracker)
    state_dict = _load_lightning_ckpt(ckpt_path)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[cluster_ply] Loaded ckpt={ckpt_path}")
    print(f"[cluster_ply] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[cluster_ply]   first missing keys: {missing[:10]}")
    if unexpected:
        print(f"[cluster_ply]   first unexpected keys: {unexpected[:10]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    wrapper.eval()

    fixed: FixedViews | None = None
    if fixed_path.exists():
        fixed = _load_fixed_views(fixed_path)
        print(f"[cluster_ply] fixed_views scene_id={fixed.scene_id} context={fixed.context} target={fixed.target}")
    elif args.use_fixed_views:
        raise FileNotFoundError(f"fixed_views.json not found at {fixed_path} but --use_fixed_views was set")

    # Find custom dataset cfg.
    custom_cfg = None
    for w in cfg.dataset:
        if hasattr(w, "custom"):
            custom_cfg = w.custom
            break
    if custom_cfg is None:
        raise ValueError("No custom dataset cfg found in cfg.dataset")

    # Build dataset; use CLI/fixed/random rules to choose scene + views.
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

    class _RandomSampler:
        def __init__(self, num_context_views: int, num_target_views: int, seed: int):
            self._num_context_views = int(num_context_views)
            self._num_target_views = int(num_target_views)
            self._seed = int(seed)

        @property
        def num_context_views(self) -> int:
            return self._num_context_views

        @property
        def num_target_views(self) -> int:
            return self._num_target_views

        def sample(self, scene: str, num_context_views: int, extrinsics, intrinsics, device=torch.device("cpu")):
            V = int(extrinsics.shape[0])
            if self._num_context_views + self._num_target_views > V:
                raise ValueError(
                    f"Requested {self._num_context_views} context and {self._num_target_views} target views, "
                    f"but scene only has {V} frames."
                )
            g = torch.Generator(device=device)
            g.manual_seed(self._seed)
            perm = torch.randperm(V, generator=g, device=device)
            ctx_idx = perm[: self._num_context_views]
            tgt_idx = perm[self._num_context_views : self._num_context_views + self._num_target_views]
            overlap = torch.tensor([0.5], device=device, dtype=torch.float32)
            return ctx_idx, tgt_idx, overlap

    # Parse optional explicit view indices from CLI.
    cli_ctx = _parse_indices_arg(args.context_views)
    cli_tgt = _parse_indices_arg(args.target_views)

    if bool(cli_ctx) != bool(cli_tgt):
        raise ValueError("Both context_views and target_views must be provided together, or both omitted.")

    # Build dataset with a temporary sampler; we'll override it below.
    ds = DatasetCustom(custom_cfg, "train", _FixedSampler([0, 1], [2]))

    # --- Choose scene ---
    if args.scene_id:
        scene_id = args.scene_id
        scene_index = _find_scene_index(ds.scenes, scene_id)
    elif args.use_fixed_views and fixed is not None:
        scene_id = fixed.scene_id
        scene_index = _find_scene_index(ds.scenes, scene_id)
    else:
        # Fully random scene driven by seed.
        rng = np.random.default_rng(int(args.seed))
        scene_index = int(rng.integers(0, len(ds.scenes)))
        scene_obj = ds.scenes[scene_index]
        scene_id = str(scene_obj.get("scene_id", scene_index))

    # --- Choose view sampler ---
    if cli_ctx and cli_tgt:
        sampler = _FixedSampler(cli_ctx, cli_tgt)
        num_ctx = len(cli_ctx)
    elif args.use_fixed_views and fixed is not None:
        if scene_id != fixed.scene_id:
            raise ValueError(
                f"--use_fixed_views requested but scene_id mismatch: "
                f"chosen scene_id={scene_id}, fixed_views.scene_id={fixed.scene_id}"
            )
        sampler = _FixedSampler(fixed.context, fixed.target)
        num_ctx = len(fixed.context)
    else:
        num_ctx = int(args.num_context_views)
        num_tgt = int(args.num_target_views)
        sampler = _RandomSampler(num_ctx, num_tgt, seed=int(args.seed))

    ds.view_sampler = sampler  # type: ignore[assignment]

    ps_h = int(custom_cfg.input_image_shape[0] // 14)
    ps_w = int(custom_cfg.input_image_shape[1] // 14)
    example = ds.getitem(scene_index, num_ctx, (ps_h, ps_w))

    # Prepare images in [0,1]
    ctx_img = example["context"]["image"].unsqueeze(0).to(device)  # [1,Vc,3,H,W]
    tgt_img = example["target"]["image"].unsqueeze(0).to(device)   # [1,Vt,3,H,W]
    images = torch.cat([ctx_img, tgt_img], dim=1)

    # Encoder forward.
    with torch.no_grad():
        enc_out = wrapper.model.encoder(images, global_step=0, visualization_dump=None)

    gaussians = enc_out.gaussians
    if enc_out.gaussian_instance_feat is None:
        raise ValueError("gaussian_instance_feat is None. Did you train with model.encoder.instance_feat_dim > 0?")

    # ---- Diagnostic prints ----
    m = gaussians.means[0]
    s = gaussians.scales[0]
    o = gaussians.opacities[0]
    print(f"[cluster_ply] means:     shape={tuple(m.shape)}, "
          f"min={m.min().item():.4f}, max={m.max().item():.4f}, "
          f"mean={m.mean().item():.4f}, std={m.std().item():.4f}")
    print(f"[cluster_ply] scales:    shape={tuple(s.shape)}, "
          f"min={s.min().item():.6f}, max={s.max().item():.6f}")
    print(f"[cluster_ply] opacities: shape={tuple(o.shape)}, "
          f"min={o.min().item():.4f}, max={o.max().item():.4f}, "
          f"mean={o.mean().item():.4f}")
    del m, s, o

    # Export a positions-only PLY for quick sanity check in MeshLab/CloudCompare.
    _pos_ply = out_dir / "positions_only.ply"
    _pos = gaussians.means[0].detach().cpu().numpy().astype(np.float32)
    from plyfile import PlyData, PlyElement

    _dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    _verts = np.empty(len(_pos), dtype=_dtype)
    _verts["x"], _verts["y"], _verts["z"] = _pos[:, 0], _pos[:, 1], _pos[:, 2]
    PlyData([PlyElement.describe(_verts, "vertex")]).write(str(_pos_ply))
    print(f"[cluster_ply] Wrote positions-only PLY: {_pos_ply}")
    del _pos, _verts

    feat_gn = enc_out.gaussian_instance_feat[0].to(device).float()  # [G,N]
    op_g = gaussians.opacities[0].to(device)
    if op_g.ndim != 1:
        op_g = op_g.reshape(-1)

    if args.cluster_algo == "kmeans":
        labels_g, centers = _cluster_gaussians(
            feat_gn,
            op_g,
            k=int(args.k),
            iters=int(args.iters),
            seed=int(args.seed),
            max_points=int(args.max_points),
            opacity_threshold=float(args.opacity_threshold),
        )
    elif args.cluster_algo == "dbscan":
        labels_g, centers = _cluster_gaussians_dbscan(
            feat_gn,
            op_g,
            eps=float(args.dbscan_eps),
            min_samples=int(args.dbscan_min_samples),
            opacity_threshold=float(args.opacity_threshold),
        )
    elif args.cluster_algo == "hdbscan":
        labels_g, centers = _cluster_gaussians_hdbscan(
            feat_gn,
            op_g,
            min_cluster_size=int(args.hdbscan_min_cluster_size),
            min_samples=int(args.hdbscan_min_samples),
            opacity_threshold=float(args.opacity_threshold),
        )
    else:  # pragma: no cover - argparse choices should prevent this.
        raise ValueError(f"Unsupported cluster_algo={args.cluster_algo}")

    n_valid = int((labels_g > 0).sum())
    n_ignored = int((labels_g == 0).sum())
    max_label = int(labels_g.max()) if labels_g.size > 0 else 0
    print(
        f"[cluster_ply] Clustered G={feat_gn.shape[0]} N={feat_gn.shape[1]} "
        f"algo={args.cluster_algo} "
        f"k_req={args.k} num_clusters={max_label}"
    )
    print(f"[cluster_ply]   valid={n_valid} ignored={n_ignored} "
          f"label range=[{labels_g.min()}, {labels_g.max()}]")

    # Labels -> palette RGB.
    lut = _make_color_lut(int(max_label) + 1, seed=int(args.palette_seed))  # 0..max_label
    rgb_u8 = lut[np.clip(labels_g, 0, int(max_label))]  # [G,3] uint8
    rgb = torch.from_numpy(rgb_u8).to(device=device, dtype=torch.float32) / 255.0  # [G,3]

    # Encode RGB into SH DC so standard GS viewers/renderers show the id-color.
    sh_dc = rgb_to_sh(rgb)  # [G,3]
    harm = torch.zeros_like(gaussians.harmonics[0]).float()  # [G,3,d_sh]
    harm[:, :, 0] = sh_dc

    out_ply = out_dir / str(args.out_name)
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        harm,
        gaussians.opacities[0],
        out_ply,
        save_sh_dc_only=True,
    )

    # Also save a simple colored point cloud (positions + cluster RGB, no GS params)
    # so you can inspect geometry in any basic viewer without GS-specific rendering.
    _cpos = gaussians.means[0].detach().cpu().numpy().astype(np.float32)
    _cdtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    _cverts = np.empty(len(_cpos), dtype=_cdtype)
    _cverts["x"], _cverts["y"], _cverts["z"] = _cpos[:, 0], _cpos[:, 1], _cpos[:, 2]
    _cverts["red"] = rgb_u8[:, 0]
    _cverts["green"] = rgb_u8[:, 1]
    _cverts["blue"] = rgb_u8[:, 2]
    _colored_ply = out_dir / "colored_pointcloud.ply"
    PlyData([PlyElement.describe(_cverts, "vertex")]).write(str(_colored_ply))
    print(f"[cluster_ply] Wrote colored point cloud: {_colored_ply}")
    del _cpos, _cverts

    # Save debug artifacts.
    np.save(out_dir / "cluster_labels.npy", labels_g.astype(np.int32))
    np.save(out_dir / "cluster_lut_u8.npy", lut.astype(np.uint8))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "ckpt": str(ckpt_path),
                "scene_id": scene_id,
                "context": example["context"]["index"].detach().cpu().tolist(),
                "target": example["target"]["index"].detach().cpu().tolist(),
                "k": int(args.k),
                "cluster_algo": str(args.cluster_algo),
                "num_clusters": int(max_label),
                "iters": int(args.iters),
                "max_points": int(args.max_points),
                "seed": int(args.seed),
                "palette_seed": int(args.palette_seed),
                "opacity_threshold": float(args.opacity_threshold),
                "out_ply": str(out_ply),
                "G": int(feat_gn.shape[0]),
                "N": int(feat_gn.shape[1]),
            },
            indent=2,
        )
    )
    print(f"[cluster_ply] Wrote {out_ply}")
    print(f"[cluster_ply] Wrote {out_dir/'cluster_labels.npy'} and {out_dir/'cluster_lut_u8.npy'}")


if __name__ == "__main__":
    _main()


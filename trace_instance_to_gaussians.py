#!/usr/bin/env python3
"""
Trace 2D instance feature maps onto pre-trained 2DGS Gaussians.

Uses diff_surfel_rasterization.trace() to back-project per-pixel instance
features (from AnySplat's instance head) onto a high-quality pre-trained
Gaussian scene, accumulating across multiple views.

Two feature-map sources:
  Mode A (online):  Run AnySplat encoder with instance head on scene images.
  Mode B (offline): Load pre-computed feature maps from .pt / .npy files.

Dependencies (anysplat conda env):
  torch, numpy, Pillow, plyfile, tqdm, diff_surfel_rasterization
  + src.misc.colmap_utils (project-local)

Example:
  # Mode B – pre-saved feature maps
  python trace_instance_to_gaussians.py \
      --source_path zipnerf/alameda \
      --ply_path zipnerf/alameda/point_cloud.ply \
      --feat_dir precomputed_feats/ \
      --feat_dim 8 --max_views 5

  # Mode A – online inference
  python trace_instance_to_gaussians.py \
      --source_path zipnerf/alameda \
      --ply_path zipnerf/alameda/point_cloud.ply \
      --run_dir output/exp_instseg_custom/2026-02-22_17-15-07 \
      --ckpt output/exp_instseg_custom/2026-02-22_17-15-07/checkpoints/last.ckpt \
      --max_views 5
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.misc.colmap_utils import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat,
)

TRACE_CHANNELS = 20  # compiled into diff_surfel_rasterization CUDA kernel


# ---------------------------------------------------------------------------
# Graphics utilities
# ---------------------------------------------------------------------------

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t):
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """OpenGL-ish projection with z_sign=1 (matches diff_surfel_rasterization)."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# ---------------------------------------------------------------------------
# Camera container
# ---------------------------------------------------------------------------

class TraceCamera:
    """Stores both COLMAP intrinsics/extrinsics and diff_surfel_rasterization matrices."""
    def __init__(self, R, T, FoVx, FoVy, image_name, width, height,
                 image_path, znear=0.01, zfar=1000.0):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.image_path = image_path
        self.znear = znear
        self.zfar = zfar

        W2C = getWorld2View2(R, T)
        self.viewmatrix = torch.tensor(W2C, device="cuda").T.contiguous()
        P = getProjectionMatrix(znear, zfar, FoVx, FoVy)
        self.projmatrix = (
            self.viewmatrix @ P.T.to("cuda").contiguous()
        ).contiguous()
        self.tanfovx = math.tan(FoVx / 2)
        self.tanfovy = math.tan(FoVy / 2)
        self.campos = torch.linalg.inv(self.viewmatrix.T)[:3, 3].cuda()


# ---------------------------------------------------------------------------
# Load Gaussians from PLY (trace-ready: linear scales, sigmoid opacity)
# ---------------------------------------------------------------------------

def load_gaussians_from_ply(ply_path, sh_degree=3):
    """Load Gaussian parameters for trace (linear-space scales, activated opacity).

    Returns scales with their original dimensionality (2 for 2DGS, 3 for 3DGS)
    in linear space, and normalized quaternions.
    """
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacities_raw = np.asarray(v["opacity"])

    # SH DC for RGB rendering during trace
    f_dc_0 = np.asarray(v["f_dc_0"])
    f_dc_1 = np.asarray(v["f_dc_1"])
    f_dc_2 = np.asarray(v["f_dc_2"])
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    colors_rgb = np.stack([
        0.5 + C0 * f_dc_0,
        0.5 + C0 * f_dc_1,
        0.5 + C0 * f_dc_2,
    ], axis=1).clip(0, 1).astype(np.float32)

    # Scales (log-space in PLY -> convert to linear)
    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_scales = len(scale_names)
    scales_log = np.zeros((xyz.shape[0], n_scales), dtype=np.float32)
    for idx, name in enumerate(scale_names):
        scales_log[:, idx] = np.asarray(v[name])

    # Rotation quaternion
    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, name in enumerate(rot_names):
        rots[:, idx] = np.asarray(v[name])

    # To GPU
    means = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    quats = torch.tensor(rots, dtype=torch.float32, device="cuda")
    colors = torch.tensor(colors_rgb, dtype=torch.float32, device="cuda")

    # Activate: exp for scales, sigmoid for opacities, normalize quaternions
    scales_log_t = torch.tensor(scales_log, dtype=torch.float32, device="cuda")
    scales_linear = torch.exp(scales_log_t)  # keep original dim (2 or 3)

    quats = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)

    opacities = torch.sigmoid(
        torch.tensor(opacities_raw, dtype=torch.float32, device="cuda")
    ).unsqueeze(-1)

    print(f"  Gaussians: {means.shape[0]:,}, scales: {n_scales}D (linear space)")
    return means, quats, scales_linear, opacities, colors


# ---------------------------------------------------------------------------
# Load COLMAP cameras
# ---------------------------------------------------------------------------

def load_colmap_cameras(source_path, images_folder, resolution):
    scene_dir = os.path.join(source_path, "sparse", "0")
    if not os.path.isdir(scene_dir):
        scene_dir = os.path.join(source_path, "sparse")
    assert os.path.isdir(scene_dir), f"sparse dir not found: {scene_dir}"

    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(scene_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(scene_dir, "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(scene_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(scene_dir, "cameras.txt"))

    images_dir = os.path.join(source_path, images_folder)
    cameras = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model in ("PINHOLE", "OPENCV"):
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        orig_w, orig_h = intr.width, intr.height
        if resolution in (1, 2, 4, 8):
            new_w, new_h = round(orig_w / resolution), round(orig_h / resolution)
        elif resolution == -1:
            if orig_w > 1600:
                s = orig_w / 1600
                new_w, new_h = int(orig_w / s), int(orig_h / s)
            else:
                new_w, new_h = orig_w, orig_h
        else:
            new_w, new_h = orig_w, orig_h

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        fx_s, fy_s = fx * scale_x, fy * scale_y

        FoVx = focal2fov(fx_s, new_w)
        FoVy = focal2fov(fy_s, new_h)

        image_path = os.path.join(images_dir, os.path.basename(extr.name))

        cam = TraceCamera(
            R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            image_name=os.path.splitext(os.path.basename(image_path))[0],
            width=new_w, height=new_h, image_path=image_path,
        )
        cameras.append(cam)

    cameras.sort(key=lambda c: c.image_name)
    return cameras


# ---------------------------------------------------------------------------
# Trace a single view
# ---------------------------------------------------------------------------

def trace_single_view(
    means, quats, scales, opacities, colors,
    img_sem, img_mask, cam, bg_color,
):
    """
    Run diff_surfel_rasterization.trace() for one camera view.

    Uses the native scales+rotations path (no cov3D_precomp).
    scales: (N, 2) for 2DGS or (N, 3) for 3DGS, in **linear** space.
    quats: (N, 4) normalized wxyz quaternions.
    img_sem: (H, W, TRACE_CHANNELS) float32 CUDA – zero-padded feature map.
    img_mask: (H, W) int32 CUDA.
    Returns (gau_sem, num_ray, radii, out_color).
    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )

    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=float(cam.tanfovx),
        tanfovy=float(cam.tanfovy),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=0,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    means2D = torch.zeros_like(means, requires_grad=False)

    with torch.no_grad():
        out_color, gau_depth, gau_sem, num_ray, radii = rasterizer.trace(
            means3D=means,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            img_sem=img_sem,
            img_mask=img_mask,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )

    return gau_sem, num_ray, radii, out_color


# ---------------------------------------------------------------------------
# Mode A: online model inference
# ---------------------------------------------------------------------------

def _load_lightning_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def load_anysplat_encoder(run_dir, ckpt_path, device="cuda"):
    """Load the AnySplat encoder (with instance head) from a training run."""
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.model import get_model
    from src.model.model_wrapper import ModelWrapper

    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    cfg_dict = OmegaConf.load(str(cfg_path))
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    model = get_model(cfg.model.encoder, cfg.model.decoder)
    wrapper = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train,
        model, get_losses(cfg.loss), StepTracker(),
    )
    state_dict = _load_lightning_ckpt(ckpt_path)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[model] Loaded ckpt (missing={len(missing)}, unexpected={len(unexpected)})")

    encoder = wrapper.model.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    feat_dim = int(getattr(encoder, "instance_feat_dim", 0))
    print(f"[model] instance_feat_dim = {feat_dim}")
    return encoder, feat_dim


def prepare_encoder_image(image_path):
    """Resize + center-crop to 448x448, return tensor in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height))
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    img = img.crop((left, top, left + 448, top + 448))

    import torchvision.transforms as T
    tensor = T.ToTensor()(img)  # [3, 448, 448] in [0, 1]
    return tensor


@torch.no_grad()
def run_encoder_batch(encoder, image_paths, device="cuda"):
    """
    Run encoder on a batch of images, return instance_feat_map.

    Returns: (feat_map, feat_dim) where feat_map is [V, D, H_feat, W_feat].
    """
    imgs = torch.stack([prepare_encoder_image(p) for p in image_paths])
    imgs = imgs.unsqueeze(0).to(device)  # [1, V, 3, 448, 448]

    enc_out = encoder(imgs, global_step=0, visualization_dump=None)
    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        raise RuntimeError(
            "Encoder returned instance_feat_map=None. "
            "Check that instance_feat_dim > 0 in the checkpoint config."
        )
    # [1, V, D, H, W] -> [V, D, H, W]
    return feat_map[0].float()


# ---------------------------------------------------------------------------
# Mode B: load pre-computed feature maps
# ---------------------------------------------------------------------------

def load_precomputed_features(feat_dir, image_names, feat_dim):
    """
    Load per-view feature maps from feat_dir/{image_name}.pt or .npy.

    Returns list of tensors [D, H_feat, W_feat] on CUDA.
    """
    feat_dir = Path(feat_dir)
    feats = []
    for name in image_names:
        pt_path = feat_dir / f"{name}.pt"
        npy_path = feat_dir / f"{name}.npy"
        if pt_path.exists():
            t = torch.load(pt_path, map_location="cuda", weights_only=True).float()
        elif npy_path.exists():
            arr = np.load(str(npy_path))
            t = torch.from_numpy(arr).float().cuda()
        else:
            raise FileNotFoundError(
                f"Feature map not found: {pt_path} or {npy_path}"
            )
        if t.dim() == 2:
            raise ValueError(f"Feature map must be 3D [D,H,W] or [H,W,D], got {t.shape}")
        if t.shape[0] != feat_dim and t.shape[-1] == feat_dim:
            t = t.permute(2, 0, 1)
        feats.append(t)
    return feats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trace 2D instance features onto pre-trained 2DGS Gaussians"
    )
    g_scene = parser.add_argument_group("Scene")
    g_scene.add_argument("--source_path", "-s", required=True)
    g_scene.add_argument("--ply_path", "-p", required=True)
    g_scene.add_argument("--images_folder", default="images")
    g_scene.add_argument("--resolution", "-r", type=int, default=1)
    g_scene.add_argument("--output_dir", "-o", default="trace_output")

    g_feat = parser.add_argument_group("Features (choose Mode A or B)")
    g_feat.add_argument("--feat_dir", default=None,
                        help="Mode B: directory of pre-computed .pt/.npy feature maps")
    g_feat.add_argument("--feat_dim", type=int, default=8,
                        help="Feature dimension (Mode B; Mode A reads from ckpt)")
    g_feat.add_argument("--run_dir", default=None,
                        help="Mode A: Hydra run directory")
    g_feat.add_argument("--ckpt", default=None,
                        help="Mode A: Lightning checkpoint path")
    g_feat.add_argument("--encoder_batch_size", type=int, default=4,
                        help="Mode A: views per encoder forward pass")

    g_run = parser.add_argument_group("Runtime")
    g_run.add_argument("--max_views", type=int, default=None)
    g_run.add_argument("--white_background", action="store_true")
    g_run.add_argument("--save_render", action="store_true",
                       help="Save per-view trace RGB renders")

    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    ply_path = os.path.abspath(args.ply_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mode_a = args.run_dir is not None and args.ckpt is not None
    mode_b = args.feat_dir is not None
    if not mode_a and not mode_b:
        parser.error("Specify --feat_dir (Mode B) or --run_dir + --ckpt (Mode A)")

    device = "cuda"
    bg_val = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    bg_color = torch.tensor(bg_val, dtype=torch.float32, device=device)

    # ---- 1. Load Gaussians ----
    means, quats, scales, opacities, colors = load_gaussians_from_ply(ply_path)
    N = means.shape[0]

    # ---- 2. Load cameras ----
    print("Loading COLMAP cameras ...")
    cameras = load_colmap_cameras(source_path, args.images_folder, args.resolution)
    print(f"  {len(cameras)} cameras loaded")

    cam_list = cameras
    if args.max_views is not None:
        cam_list = cameras[:args.max_views]
        print(f"  Using first {len(cam_list)} views")

    # ---- 3. Prepare feature source ----
    encoder = None
    feat_dim = args.feat_dim

    if mode_a:
        print("\n[Mode A] Loading AnySplat encoder ...")
        encoder, feat_dim = load_anysplat_encoder(args.run_dir, args.ckpt, device)
        print(f"  Feature dim = {feat_dim}")

    if feat_dim > TRACE_CHANNELS:
        raise ValueError(
            f"feat_dim={feat_dim} > TRACE_CHANNELS={TRACE_CHANNELS}. "
            f"Cannot trace more than {TRACE_CHANNELS} channels."
        )

    if mode_b:
        print(f"\n[Mode B] Loading pre-computed features from {args.feat_dir}")
        precomp_feats = load_precomputed_features(
            args.feat_dir,
            [c.image_name for c in cam_list],
            feat_dim,
        )
        print(f"  Loaded {len(precomp_feats)} feature maps, dim={feat_dim}")

    if args.save_render:
        os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)

    # ---- 4/5. Trace loop ----
    sum_gau_sem = torch.zeros(N, feat_dim, device=device, dtype=torch.float32)
    sum_num_ray = torch.zeros(N, device=device, dtype=torch.float32)

    # For Mode A: process views through encoder in batches
    if mode_a:
        batch_size = args.encoder_batch_size
        all_feat_maps = []
        image_paths = [c.image_path for c in cam_list]

        print(f"\n[Mode A] Running encoder on {len(cam_list)} views "
              f"(batch_size={batch_size}) ...")
        for start in range(0, len(cam_list), batch_size):
            end = min(start + batch_size, len(cam_list))
            batch_paths = image_paths[start:end]
            feat_batch = run_encoder_batch(encoder, batch_paths, device)
            for vi in range(feat_batch.shape[0]):
                all_feat_maps.append(feat_batch[vi])  # [D, H_feat, W_feat]

        del encoder
        torch.cuda.empty_cache()

    print(f"\nTracing {len(cam_list)} views (feat_dim={feat_dim}, "
          f"TRACE_CHANNELS={TRACE_CHANNELS}) ...")

    for idx, cam in enumerate(tqdm(cam_list, desc="Trace")):
        H, W = cam.image_height, cam.image_width

        # Get feature map for this view
        if mode_a:
            feat_2d = all_feat_maps[idx]  # [D, H_feat, W_feat]
        else:
            feat_2d = precomp_feats[idx]  # [D, H_feat, W_feat]

        # Resize feature map to match trace rendering resolution
        if feat_2d.shape[1] != H or feat_2d.shape[2] != W:
            feat_2d = F.interpolate(
                feat_2d.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # Pad to TRACE_CHANNELS: [D, H, W] -> [H, W, TRACE_CHANNELS]
        feat_hwc = feat_2d.permute(1, 2, 0).contiguous()  # [H, W, D]
        if feat_dim < TRACE_CHANNELS:
            pad = torch.zeros(
                H, W, TRACE_CHANNELS - feat_dim,
                device=device, dtype=torch.float32,
            )
            feat_hwc = torch.cat([feat_hwc, pad], dim=2)

        img_mask = torch.ones(H, W, dtype=torch.int32, device=device)

        gau_sem, num_ray, radii, out_color = trace_single_view(
            means, quats, scales, opacities, colors,
            feat_hwc, img_mask, cam, bg_color,
        )

        # Accumulate (only first feat_dim channels)
        sum_gau_sem += gau_sem[:, :feat_dim]
        sum_num_ray += num_ray.float()

        if args.save_render:
            render_np = (out_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
                         ).astype(np.uint8)
            Image.fromarray(render_np).save(
                os.path.join(output_dir, "renders", f"{cam.image_name}.png")
            )

    # ---- 6. Normalize and save ----
    valid_mask = sum_num_ray > 0
    final_feat = torch.zeros_like(sum_gau_sem)
    final_feat[valid_mask] = (
        sum_gau_sem[valid_mask] / sum_num_ray[valid_mask].unsqueeze(-1)
    )

    n_valid = valid_mask.sum().item()
    print(f"\n========== Results ==========")
    print(f"  Gaussians total:   {N:,}")
    print(f"  Gaussians traced:  {n_valid:,} ({100 * n_valid / N:.1f}%)")
    print(f"  Feature dim:       {feat_dim}")
    print(f"  Views used:        {len(cam_list)}")

    # Save
    out_path_pt = os.path.join(output_dir, "gaussian_instance_feat.pt")
    torch.save({
        "feat": final_feat.cpu(),
        "num_ray": sum_num_ray.cpu(),
        "feat_unnorm": sum_gau_sem.cpu(),
        "feat_dim": feat_dim,
        "n_views": len(cam_list),
        "ply_path": ply_path,
        "source_path": source_path,
    }, out_path_pt)
    print(f"  Saved to {out_path_pt}")

    out_path_npy = os.path.join(output_dir, "gaussian_instance_feat.npy")
    np.save(out_path_npy, final_feat.cpu().numpy())
    print(f"  Saved to {out_path_npy}")

    # Stats
    feat_norms = final_feat[valid_mask].norm(dim=1)
    print(f"\n  Feature stats (valid Gaussians):")
    print(f"    norm  min={feat_norms.min():.4f}  mean={feat_norms.mean():.4f}  "
          f"max={feat_norms.max():.4f}")
    print(f"    num_ray  min={sum_num_ray[valid_mask].min():.0f}  "
          f"mean={sum_num_ray[valid_mask].mean():.1f}  "
          f"max={sum_num_ray[valid_mask].max():.0f}")


if __name__ == "__main__":
    main()

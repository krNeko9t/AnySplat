#!/usr/bin/env python3
"""
Standalone render-and-compare script for AnySplat.
Loads COLMAP cameras + a trained 2DGS surfel PLY, renders each view,
and compares against ground-truth images (PSNR / L1).

Uses gsplat for rasterisation (same renderer as AnySplat).

Dependencies (all available in the anysplat conda env):
  torch, numpy, Pillow, plyfile, tqdm, gsplat
  + src.misc.colmap_utils (project-local)
"""

import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData
from gsplat import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.misc.colmap_utils import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat,
)


# ---------------------------------------------------------------------------
# Graphics utilities
# ---------------------------------------------------------------------------

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def getWorld2View2(R, t):
    """Build 4x4 W2C matrix from COLMAP-convention R (stored transposed) and T."""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# ---------------------------------------------------------------------------
# Camera info container
# ---------------------------------------------------------------------------

class CameraView:
    def __init__(self, R, T, FoVx, FoVy, fx, fy, cx, cy,
                 image, image_name, width, height):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.original_image = image  # [3, H, W] float32, 0-1

        # W2C 4x4 matrix for gsplat
        self.w2c = torch.tensor(getWorld2View2(R, T), device="cuda").unsqueeze(0)  # [1, 4, 4]

        # Intrinsic 3x3 for gsplat (pixel units)
        K = torch.zeros(1, 3, 3, device="cuda")
        K[0, 0, 0] = fx
        K[0, 1, 1] = fy
        K[0, 0, 2] = cx
        K[0, 1, 2] = cy
        K[0, 2, 2] = 1.0
        self.K = K


# ---------------------------------------------------------------------------
# Load trained Gaussians from PLY
# ---------------------------------------------------------------------------

def load_gaussians_from_ply(ply_path, sh_degree=3):
    """Return Gaussian parameters ready for gsplat rasterisation."""
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacities = np.asarray(v["opacity"])

    # SH DC
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(v["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(v["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(v["f_dc_2"])

    # SH rest
    extra_f_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_rest_expected = 3 * (sh_degree + 1) ** 2 - 3
    assert len(extra_f_names) == n_rest_expected, (
        f"Expected {n_rest_expected} f_rest fields, got {len(extra_f_names)}"
    )
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(v[name])
    features_extra = features_extra.reshape(
        (xyz.shape[0], 3, (sh_degree + 1) ** 2 - 1)
    )

    # Scale (2D surfel: 2 scales in log-space; pad third to near-zero)
    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_scales = len(scale_names)
    scales_raw = np.zeros((xyz.shape[0], n_scales))
    for idx, name in enumerate(scale_names):
        scales_raw[:, idx] = np.asarray(v[name])

    if n_scales == 2:
        pad = np.full((xyz.shape[0], 1), -20.0, dtype=np.float32)
        scales_raw = np.concatenate([scales_raw, pad], axis=1)

    # Rotation quaternion
    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, name in enumerate(rot_names):
        rots[:, idx] = np.asarray(v[name])

    # To GPU tensors (gsplat expects raw log-scales and un-activated opacity)
    means = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    opacities_raw = torch.tensor(opacities, dtype=torch.float32, device="cuda")
    scales_log = torch.tensor(scales_raw, dtype=torch.float32, device="cuda")
    quats = torch.tensor(rots, dtype=torch.float32, device="cuda")

    # gsplat applies sigmoid to opacities and exp to scales internally,
    # so we pass raw (pre-activation) values
    opacity_activated = torch.sigmoid(opacities_raw)

    # SH features: gsplat expects [N, K, 3] where K = (sh_degree+1)^2
    # dc: [N, 3, 1] → [N, 1, 3], rest: [N, 3, 15] → [N, 15, 3]
    dc = torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous()
    rest = torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous()
    colors = torch.cat([dc, rest], dim=1)  # [N, 16, 3]

    print(f"  Gaussians: {means.shape[0]:,}, SH degree: {sh_degree}, "
          f"raw scales: {n_scales} (padded to 3)")
    return means, quats, scales_log, opacity_activated, colors, sh_degree


# ---------------------------------------------------------------------------
# Render one view with gsplat
# ---------------------------------------------------------------------------

def render_view(cam, means, quats, scales_log, opacity, colors, sh_degree, bg_color):
    W, H = cam.image_width, cam.image_height
    scales = torch.exp(scales_log)

    rendering, alpha, _ = rasterization(
        means, quats, scales, opacity, colors,
        cam.w2c, cam.K, W, H,
        sh_degree=sh_degree,
        render_mode="RGB",
        packed=False,
        near_plane=0.01,
        far_plane=1000.0,
        backgrounds=bg_color.unsqueeze(0),
        rasterize_mode="classic",
    )
    # rendering: [1, H, W, 3]
    return rendering.squeeze(0).permute(2, 0, 1)  # [3, H, W]


# ---------------------------------------------------------------------------
# COLMAP scene loading → list[CameraView]
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
        h, w = intr.height, intr.width

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

        FoVx = focal2fov(fx, w)
        FoVy = focal2fov(fy, h)

        image_path = os.path.join(images_dir, os.path.basename(extr.name))
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        pil_img = Image.open(image_path)
        orig_w, orig_h = pil_img.size

        # Resolution handling
        if resolution in (1, 2, 4, 8):
            new_w, new_h = round(orig_w / resolution), round(orig_h / resolution)
        elif resolution == -1:
            if orig_w > 1600:
                scale = orig_w / 1600
                new_w, new_h = int(orig_w / scale), int(orig_h / scale)
            else:
                new_w, new_h = orig_w, orig_h
        else:
            new_w, new_h = orig_w, orig_h

        # Scale intrinsics to match resolution
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        if (new_w, new_h) != (orig_w, orig_h):
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        img_tensor = torch.from_numpy(
            np.array(pil_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)[:3]

        cam = CameraView(
            R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            fx=fx_scaled, fy=fy_scaled, cx=cx_scaled, cy=cy_scaled,
            image=img_tensor, image_name=image_name,
            width=new_w, height=new_h,
        )
        cameras.append(cam)

    cameras.sort(key=lambda c: c.image_name)
    return cameras


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return torch.tensor(100.0)
    return 10.0 * torch.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render with COLMAP cameras + trained surfel PLY, compare to GT"
    )
    parser.add_argument("--source_path", "-s", required=True,
                        help="Scene root with sparse/ and images/")
    parser.add_argument("--ply_path", "-p", required=True,
                        help="Trained point_cloud.ply")
    parser.add_argument("--output_dir", "-o", default="render_compare",
                        help="Output directory")
    parser.add_argument("--resolution", "-r", type=int, default=1,
                        help="Resolution divisor (1/2/4/8) or -1 for auto")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--images_folder", default="images")
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    ply_path = os.path.abspath(args.ply_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.isdir(source_path), f"source_path not found: {source_path}"
    assert os.path.isfile(ply_path), f"ply_path not found: {ply_path}"

    # 1. Load cameras
    print("Loading COLMAP cameras ...")
    cameras = load_colmap_cameras(source_path, args.images_folder, args.resolution)
    print(f"  {len(cameras)} cameras loaded")

    # 2. Load Gaussians
    means, quats, scales_log, opacity, colors, sh_deg = load_gaussians_from_ply(
        ply_path, sh_degree=args.sh_degree
    )

    # 3. Prepare
    bg_val = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_val, dtype=torch.float32, device="cuda")

    cam_list = cameras
    if args.max_views is not None:
        cam_list = cameras[: args.max_views]
        print(f"  Rendering first {len(cam_list)} views (--max_views={args.max_views})")

    for sub in ("renders", "gt", "compare"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    psnr_list, l1_list = [], []

    print("Rendering and comparing ...")
    for idx, cam in enumerate(tqdm(cam_list, desc="Render")):
        with torch.no_grad():
            rend = render_view(
                cam, means, quats, scales_log, opacity, colors, sh_deg, background
            )
            rend = torch.clamp(rend, 0.0, 1.0)

        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)

        if rend.shape != gt.shape:
            rend = F.interpolate(
                rend.unsqueeze(0), size=(gt.shape[1], gt.shape[2]),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        p = psnr(rend, gt).item()
        l1 = (rend - gt).abs().mean().item()
        psnr_list.append(p)
        l1_list.append(l1)

        name = cam.image_name
        rend_np = (rend.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rend_np).save(os.path.join(output_dir, "renders", f"{name}.png"))
        Image.fromarray(gt_np).save(os.path.join(output_dir, "gt", f"{name}.png"))
        compare = np.concatenate([gt_np, rend_np], axis=1)
        Image.fromarray(compare).save(
            os.path.join(output_dir, "compare", f"{name}_gt_vs_render.png")
        )

    psnr_mean = float(np.mean(psnr_list))
    l1_mean = float(np.mean(l1_list))

    print("\n========== Results ==========")
    print(f"  Views:     {len(cam_list)}")
    print(f"  Mean PSNR: {psnr_mean:.2f} dB")
    print(f"  Mean L1:   {l1_mean:.6f}")
    print(f"  Output:    {output_dir}")

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"source_path = {source_path}\n")
        f.write(f"ply_path = {ply_path}\n")
        f.write(f"num_views = {len(cam_list)}\n")
        f.write(f"mean_PSNR = {psnr_mean:.4f}\n")
        f.write(f"mean_L1 = {l1_mean:.6f}\n\n")
        f.write("per_view:\n")
        for i, cam in enumerate(cam_list):
            f.write(f"  {cam.image_name}  PSNR={psnr_list[i]:.2f}  L1={l1_list[i]:.6f}\n")
    print(f"  Metrics saved to {output_dir}/metrics.txt")

    if psnr_mean < 15.0:
        print("\n[WARN] Mean PSNR is low — camera/PLY coordinate mismatch likely.")
    else:
        print("\n[OK] Rendering matches GT well.")


if __name__ == "__main__":
    main()

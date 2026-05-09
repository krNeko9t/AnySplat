"""Load TraceCamera list from COLMAP sparse reconstruction."""

from __future__ import annotations

import os

import numpy as np
import torch

from src.coord import (
    CameraConvention,
    CameraPose,
    ExtrinsicType,
    colmap_to_opencv_intrinsics,
    log_coordinate_op,
)
from src.misc.colmap_utils import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)

from .trace_camera import TraceCamera, focal2fov


def load_colmap_cameras(source_path, images_folder, resolution):
    """Load cameras from COLMAP ``sparse/`` (binary or text).

    Extrinsics are built with :meth:`CameraPose.from_qvec_tvec` (COLMAP ``qvec``,
    ``tvec`` → world-to-camera). ``R``, ``T`` stored on :class:`TraceCamera` are
    the decomposition used by :func:`w2c_from_colmap_rt`. Intrinsics are
    converted COLMAP → OpenCV pixel centre via ``colmap_to_opencv_intrinsics``.
    """
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

        qvec = torch.tensor(extr.qvec, dtype=torch.float64)
        tvec = torch.tensor(extr.tvec, dtype=torch.float64)
        pose = CameraPose.from_qvec_tvec(
            qvec, tvec, CameraConvention.COLMAP, ExtrinsicType.W2C
        )
        w2c_np = pose.w2c.detach().cpu().numpy().astype(np.float32)
        r_mat = w2c_np[:3, :3].T
        t_vec = w2c_np[:3, 3]

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model in ("PINHOLE", "OPENCV"):
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        k_o = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        k_o = colmap_to_opencv_intrinsics(k_o)
        fx, fy = float(k_o[0, 0]), float(k_o[1, 1])
        cx, cy = float(k_o[0, 2]), float(k_o[1, 2])

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

        fov_x = focal2fov(fx_s, new_w)
        fov_y = focal2fov(fy_s, new_h)

        image_path = os.path.join(images_dir, os.path.basename(extr.name))

        cam = TraceCamera(
            R=r_mat,
            T=t_vec,
            FoVx=fov_x,
            FoVy=fov_y,
            image_name=os.path.splitext(os.path.basename(image_path))[0],
            width=new_w,
            height=new_h,
            image_path=image_path,
        )
        cam.fx_raw, cam.fy_raw = fx, fy
        cam.cx_raw, cam.cy_raw = cx, cy
        cam.orig_width, cam.orig_height = orig_w, orig_h
        cameras.append(cam)

    cameras.sort(key=lambda c: c.image_name)
    log_coordinate_op(
        "load_colmap_cameras",
        CameraConvention.COLMAP,
        ExtrinsicType.W2C,
        (len(cameras), 4, 4),
        context=f"sparse={scene_dir} images_folder={images_folder}",
    )
    return cameras

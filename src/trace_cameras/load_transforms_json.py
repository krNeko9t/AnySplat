"""Load TraceCamera list from NeRF-style transforms JSON (e.g. transforms_train.json).

World-frame translations in ``transform_matrix`` must match the Gaussian PLY / scene.
The JSON ``camera_model`` field usually names distortion/intrinsics family (e.g. OPENCV),
not the extrinsic axis convention of ``transform_matrix``.

Typical exports store ``transform_matrix`` as camera-to-world in an OpenGL-like axis
group (see ``CameraConvention.NERFSTUDIO``). Use ``axis_convention="opencv"`` when your
matrices are already OpenCV camera axes + c2w without an extra flip.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.coord import CameraConvention, CameraPose, ExtrinsicType, log_coordinate_op

from .trace_camera import TraceCamera, focal2fov, scaled_image_size


def _axis_convention_from_string(name: str) -> CameraConvention:
    n = name.lower().strip()
    if n in ("nerfstudio", "blender", "opengl"):
        return CameraConvention.NERFSTUDIO
    if n == "opencv":
        return CameraConvention.OPENCV
    raise ValueError(
        f"Unknown transforms_axis '{name}', expected nerfstudio|opengl|blender|opencv"
    )


def load_transforms_json_cameras(
    transforms_path: str,
    resolution: int,
    transforms_axis: str = "nerfstudio",
) -> list[TraceCamera]:
    """Pinhole cameras from transforms JSON; distortion ignored (matches trace pipeline).

    Parameters
    ----------
    transforms_path
        Path to ``transforms_train.json`` / ``transforms.json``.
    resolution
        Same semantics as COLMAP loader (1/2/4/8, -1 for auto cap at ~1600px wide).
    transforms_axis
        ``nerfstudio`` (default): interpret ``transform_matrix`` as OpenGL-like c2w,
        convert via ``CameraPose.to(OPENCV, W2C)``. ``opencv``: matrices already
        OpenCV-axis c2w — only invert to w2c for TraceCamera.
    """
    axis_conv = _axis_convention_from_string(transforms_axis)
    json_dir = os.path.dirname(os.path.abspath(transforms_path))

    with open(transforms_path, encoding="utf-8") as f:
        data = json.load(f)

    fx = float(data["fl_x"])
    fy = float(data["fl_y"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    orig_w = int(data["w"])
    orig_h = int(data["h"])

    new_w, new_h = scaled_image_size(orig_w, orig_h, resolution)
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    fx_s, fy_s = fx * scale_x, fy * scale_y
    cx_s, cy_s = cx * scale_x, cy * scale_y

    cameras: list[TraceCamera] = []
    for frame in data["frames"]:
        mat = torch.tensor(
            np.asarray(frame["transform_matrix"], dtype=np.float64),
            dtype=torch.float64,
        ).unsqueeze(0)
        pose = CameraPose.from_matrix(mat, axis_conv, ExtrinsicType.C2W)
        w2c = pose.to(CameraConvention.OPENCV, ExtrinsicType.W2C).matrix.squeeze(0)
        w2c_np = w2c.detach().cpu().numpy().astype(np.float32)
        r_mat = w2c_np[:3, :3].T
        t_vec = w2c_np[:3, 3]

        rel = frame["file_path"].replace("\\", "/").lstrip("./")
        image_path = os.path.normpath(os.path.join(json_dir, rel))

        fov_x = focal2fov(fx_s, new_w)
        fov_y = focal2fov(fy_s, new_h)

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
        "load_transforms_json_cameras",
        axis_conv,
        ExtrinsicType.C2W,
        (len(cameras), 4, 4),
        context=f"json={transforms_path} transforms_axis={transforms_axis}",
    )
    return cameras


def resolve_transforms_json_path(source_path: str, explicit: str | None) -> str:
    """Return absolute path to transforms JSON.

    If ``explicit`` is set and is a file, use it. Otherwise try
    ``transforms_train.json`` then ``transforms.json`` under ``source_path``.
    """
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"--transforms_json not found: {p}")

    root = os.path.abspath(os.path.expanduser(source_path))
    for name in ("transforms_train.json", "transforms.json"):
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "No transforms JSON found. Pass --transforms_json or place "
        "transforms_train.json / transforms.json under source_path."
    )

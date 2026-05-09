"""TraceCamera + projection helpers for diff_surfel_rasterization."""

from __future__ import annotations

import math

import numpy as np
import torch


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def w2c_from_colmap_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 world-to-camera from COLMAP-style ``(R, t)`` used by :class:`TraceCamera`.

    ``R`` is ``(qvec2rotmat(qvec)).T`` and ``t`` is ``tvec``, consistent with
    :meth:`CameraPose.from_qvec_tvec` after decomposing ``pose.w2c`` into these
    components.
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def get_projection_matrix(znear, zfar, fovX, fovY):
    """OpenGL-ish projection with z_sign=1 (matches diff_surfel_rasterization)."""
    tan_half_fov_y = math.tan(fovY / 2)
    tan_half_fov_x = math.tan(fovX / 2)
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right
    p = torch.zeros(4, 4)
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def scaled_image_size(orig_w: int, orig_h: int, resolution: int) -> tuple[int, int]:
    """Match COLMAP loader behaviour for ``resolution``."""
    if resolution in (1, 2, 4, 8):
        return round(orig_w / resolution), round(orig_h / resolution)
    if resolution == -1:
        if orig_w > 1600:
            s = orig_w / 1600
            return int(orig_w / s), int(orig_h / s)
        return orig_w, orig_h
    return orig_w, orig_h


class TraceCamera:
    """Stores both COLMAP intrinsics/extrinsics and diff_surfel_rasterization matrices."""

    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image_name,
        width,
        height,
        image_path,
        znear=0.01,
        zfar=1000.0,
    ):
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

        w2c = w2c_from_colmap_rt(R, T)
        self.viewmatrix = torch.tensor(w2c, device="cuda").T.contiguous()
        p_mat = get_projection_matrix(znear, zfar, FoVx, FoVy)
        self.projmatrix = (
            self.viewmatrix @ p_mat.T.to("cuda").contiguous()
        ).contiguous()
        self.tanfovx = math.tan(FoVx / 2)
        self.tanfovy = math.tan(FoVy / 2)
        self.campos = torch.linalg.inv(self.viewmatrix.T)[:3, 3].cuda()

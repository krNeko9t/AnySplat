"""
Internal conversion primitives for camera poses.

Public code should use ``CameraPose.to()`` instead of calling these directly.
Only ``se3_inv`` and ``qvec_to_rotmat`` are re-exported from ``src.coord``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .conventions import AXIS_FLIP, AXIS_GROUP, CameraConvention, ExtrinsicType


# ---------------------------------------------------------------------------
# Camera-axis conversion
# ---------------------------------------------------------------------------

def convert_axes(
    mat: Tensor,
    src: CameraConvention,
    dst: CameraConvention,
    extrinsic_type: ExtrinsicType,
) -> Tensor:
    """Convert camera-axis convention.  Same group → no-op."""
    if AXIS_GROUP[src] == AXIS_GROUP[dst]:
        return mat
    flip = AXIS_FLIP.to(device=mat.device, dtype=mat.dtype)
    if extrinsic_type == ExtrinsicType.C2W:
        return mat @ flip
    else:
        return flip @ mat


# ---------------------------------------------------------------------------
# SE(3) inverse
# ---------------------------------------------------------------------------

def se3_inv(mat: Tensor) -> Tensor:
    """Efficient SE(3) inverse using the R^T / -R^T t structure.

    Numerically more stable than ``torch.inverse`` for rigid transforms.
    Supports batched input of shape ``(*, 4, 4)`` or ``(*, 3, 4)``.
    Always returns ``(*, 4, 4)``.
    """
    R = mat[..., :3, :3]
    t = mat[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t)
    bottom = mat.new_zeros(mat.shape[:-2] + (1, 4))
    bottom[..., 0, 3] = 1.0
    top = torch.cat([R_inv, t_inv], dim=-1)
    return torch.cat([top, bottom], dim=-2)


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: Tensor) -> Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Supports batched input ``(*, 4) -> (*, 3, 3)``.
    """
    qvec = qvec / (qvec.norm(dim=-1, keepdim=True) + 1e-12)
    w, x, y, z = qvec.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w),
            2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(qvec.shape[:-1] + (3, 3))
    return R

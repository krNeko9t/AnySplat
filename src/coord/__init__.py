"""
Unified coordinate-system utilities for AnySplat.

Quick-start
-----------
>>> from src.coord import CameraPose, CameraConvention, ExtrinsicType
>>> pose = CameraPose.from_matrix(mat, CameraConvention.BLENDER, ExtrinsicType.C2W)
>>> opencv_c2w = pose.to(CameraConvention.OPENCV).matrix

Convention reference
--------------------
OpenCV / COLMAP  : X-right, Y-down,  Z-forward    (camera looks +Z)
OpenGL / Blender : X-right, Y-up,    Z-backward   (camera looks -Z)
"""

from .camera_pose import CameraPose
from .conventions import (
    AXIS_FLIP,
    AXIS_GROUP,
    CameraConvention,
    ExtrinsicType,
    IntrinsicConvention,
)
from .conversions import qvec_to_rotmat, se3_inv
from .intrinsics import (
    colmap_to_opencv_intrinsics,
    convert_intrinsics,
    opencv_to_colmap_intrinsics,
)
from .ply_io import GaussianPlyData, export_gaussian_ply, load_gaussian_ply
from ._logging import assert_convention, log_coordinate_op

__all__ = [
    "CameraPose",
    "CameraConvention",
    "ExtrinsicType",
    "IntrinsicConvention",
    "AXIS_FLIP",
    "AXIS_GROUP",
    "se3_inv",
    "qvec_to_rotmat",
    "convert_intrinsics",
    "colmap_to_opencv_intrinsics",
    "opencv_to_colmap_intrinsics",
    "log_coordinate_op",
    "assert_convention",
    "GaussianPlyData",
    "export_gaussian_ply",
    "load_gaussian_ply",
]

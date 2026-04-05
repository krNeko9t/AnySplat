"""
Intrinsic-matrix conversion utilities.

Handles the pixel-center offset between COLMAP (0.5, 0.5) and OpenCV (0, 0).
Supports both numpy arrays and torch tensors.
"""

from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from .conventions import IntrinsicConvention

K = TypeVar("K", Tensor, np.ndarray)


def convert_intrinsics(
    intrinsic: K,
    src: IntrinsicConvention,
    dst: IntrinsicConvention,
) -> K:
    """Convert intrinsic matrix between pixel-center conventions.

    Parameters
    ----------
    intrinsic : Tensor | ndarray
        Intrinsic matrix of shape ``(*, 3, 3)``.
    src, dst : IntrinsicConvention
        Source and destination pixel-center conventions.

    Returns
    -------
    Same type and shape as *intrinsic*, with cx/cy adjusted.
    """
    if src == dst:
        return intrinsic

    # COLMAP (0.5, 0.5) -> OpenCV (0, 0): subtract 0.5
    # OpenCV (0, 0) -> COLMAP (0.5, 0.5): add 0.5
    if src == IntrinsicConvention.COLMAP and dst == IntrinsicConvention.OPENCV:
        delta = -0.5
    else:
        delta = 0.5

    if isinstance(intrinsic, np.ndarray):
        out = intrinsic.copy()
        out[..., 0, 2] += delta
        out[..., 1, 2] += delta
        return out
    else:
        out = intrinsic.clone()
        out[..., 0, 2] += delta
        out[..., 1, 2] += delta
        return out


# Backward-compatible aliases for the two common directions.

def colmap_to_opencv_intrinsics(K_mat: K) -> K:
    """Shorthand: COLMAP → OpenCV intrinsic (cx, cy -= 0.5)."""
    return convert_intrinsics(K_mat, IntrinsicConvention.COLMAP, IntrinsicConvention.OPENCV)


def opencv_to_colmap_intrinsics(K_mat: K) -> K:
    """Shorthand: OpenCV → COLMAP intrinsic (cx, cy += 0.5)."""
    return convert_intrinsics(K_mat, IntrinsicConvention.OPENCV, IntrinsicConvention.COLMAP)

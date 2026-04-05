"""Spherical-harmonic helpers for 3D Gaussian / NeRF-style RGB."""

from __future__ import annotations

from torch import Tensor

# Y_0^0 normalisation constant: 1 / (2 * sqrt(pi))
SH_DC_C0 = 0.28209479177387814


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Map RGB in [0, 1] to degree-0 SH coefficients (DC band only)."""
    return (rgb - 0.5) / SH_DC_C0

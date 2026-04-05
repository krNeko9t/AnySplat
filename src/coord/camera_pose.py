"""
CameraPose — a 4x4 SE(3) matrix with coordinate-convention metadata.

All convention conversions go through ``CameraPose.to()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .conventions import CameraConvention, ExtrinsicType
from .conversions import convert_axes, qvec_to_rotmat, se3_inv


@dataclass
class CameraPose:
    """A camera extrinsic matrix together with its coordinate-system metadata.

    Parameters
    ----------
    matrix : Tensor
        Batched SE(3) matrices of shape ``(*, 4, 4)``.
    convention : CameraConvention
        Which camera-axis convention the matrix is expressed in.
    extrinsic_type : ExtrinsicType
        Whether the matrix maps camera→world (c2w) or world→camera (w2c).
    """

    matrix: Tensor
    convention: CameraConvention
    extrinsic_type: ExtrinsicType

    # ------------------------------------------------------------------
    # The single conversion entry-point
    # ------------------------------------------------------------------

    def to(
        self,
        convention: Optional[CameraConvention] = None,
        extrinsic_type: Optional[ExtrinsicType] = None,
    ) -> "CameraPose":
        """Convert to a different convention and/or extrinsic type.

        Omitted arguments keep their current value.

        Examples
        --------
        >>> pose.to(CameraConvention.OPENCV)             # axis change only
        >>> pose.to(extrinsic_type=ExtrinsicType.W2C)     # flip c2w↔w2c only
        >>> pose.to(CameraConvention.OPENCV, ExtrinsicType.W2C)  # both
        """
        tgt_conv = convention if convention is not None else self.convention
        tgt_type = extrinsic_type if extrinsic_type is not None else self.extrinsic_type

        mat = self.matrix

        # Step 1: convert camera axes (must happen while we know current ext type)
        if tgt_conv != self.convention:
            mat = convert_axes(mat, self.convention, tgt_conv, self.extrinsic_type)

        # Step 2: flip extrinsic direction
        if tgt_type != self.extrinsic_type:
            mat = se3_inv(mat)

        return CameraPose(matrix=mat, convention=tgt_conv, extrinsic_type=tgt_type)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def c2w(self) -> Tensor:
        """Return the c2w matrix (in the current axis convention)."""
        if self.extrinsic_type == ExtrinsicType.C2W:
            return self.matrix
        return se3_inv(self.matrix)

    @property
    def w2c(self) -> Tensor:
        """Return the w2c matrix (in the current axis convention)."""
        if self.extrinsic_type == ExtrinsicType.W2C:
            return self.matrix
        return se3_inv(self.matrix)

    def inverse(self) -> "CameraPose":
        """Efficient SE(3) inverse.  Flips the extrinsic_type label."""
        flipped = (
            ExtrinsicType.W2C
            if self.extrinsic_type == ExtrinsicType.C2W
            else ExtrinsicType.C2W
        )
        return CameraPose(
            matrix=se3_inv(self.matrix),
            convention=self.convention,
            extrinsic_type=flipped,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_matrix(
        mat: Tensor,
        convention: CameraConvention,
        extrinsic_type: ExtrinsicType,
    ) -> "CameraPose":
        """Wrap an existing ``(*, 4, 4)`` matrix."""
        if mat.shape[-2:] != (4, 4):
            raise ValueError(f"Expected (*, 4, 4) matrix, got shape {mat.shape}")
        return CameraPose(matrix=mat, convention=convention, extrinsic_type=extrinsic_type)

    @staticmethod
    def from_Rt(
        R: Tensor,
        t: Tensor,
        convention: CameraConvention,
        extrinsic_type: ExtrinsicType,
    ) -> "CameraPose":
        """Build a 4x4 SE(3) from ``(*, 3, 3)`` R and ``(*, 3)`` t."""
        mat = torch.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype, device=R.device)
        mat[..., :3, :3] = R
        mat[..., :3, 3] = t
        mat[..., 3, 3] = 1.0
        return CameraPose(matrix=mat, convention=convention, extrinsic_type=extrinsic_type)

    @staticmethod
    def from_qvec_tvec(
        qvec: Tensor,
        tvec: Tensor,
        convention: CameraConvention = CameraConvention.COLMAP,
        extrinsic_type: ExtrinsicType = ExtrinsicType.W2C,
    ) -> "CameraPose":
        """Build from COLMAP-style quaternion ``(w,x,y,z)`` + translation."""
        R = qvec_to_rotmat(qvec)
        return CameraPose.from_Rt(R, tvec, convention, extrinsic_type)

    @staticmethod
    def from_pt3d(R: Tensor, T: Tensor) -> "CameraPose":
        """Build from PyTorch3D (R, T) convention.

        PT3D stores the rotation transposed relative to standard convention
        and flips the X/Y axes.  This method normalises to OpenCV c2w.
        """
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float()

        # PT3D: R_pt3d is the transpose of the standard rotation;
        # X and Y axes are negated relative to OpenCV.
        # opencv_R = diag(-1,-1,1) @ R_pt3d^T
        # opencv_t = diag(-1,-1,1) @ T_pt3d
        flip = torch.tensor([-1.0, -1.0, 1.0], dtype=R.dtype, device=R.device)
        R_opencv = R.transpose(-1, -2) * flip[..., :, None]  # (..., 3, 3)
        t_opencv = T * flip
        return CameraPose.from_Rt(R_opencv, t_opencv, CameraConvention.OPENCV, ExtrinsicType.C2W)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CameraPose({self.convention.value}, {self.extrinsic_type.value}, "
            f"shape={list(self.matrix.shape)})"
        )

"""
Unit tests for the src.coord coordinate-system module.

Run with:  python -m pytest tests/test_coord.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.coord import (
    AXIS_FLIP,
    CameraConvention,
    CameraPose,
    ExtrinsicType,
    GaussianPlyData,
    IntrinsicConvention,
    colmap_to_opencv_intrinsics,
    convert_intrinsics,
    export_gaussian_ply,
    load_gaussian_ply,
    opencv_to_colmap_intrinsics,
    qvec_to_rotmat,
    se3_inv,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _random_se3(batch: int = 1) -> torch.Tensor:
    """Generate random valid SE(3) matrices."""
    mat = torch.zeros(batch, 4, 4)
    for i in range(batch):
        q = torch.randn(4)
        q = q / q.norm()
        R = qvec_to_rotmat(q.unsqueeze(0)).squeeze(0)
        mat[i, :3, :3] = R
        mat[i, :3, 3] = torch.randn(3)
        mat[i, 3, 3] = 1.0
    return mat


ALL_CONVENTIONS = list(CameraConvention)


# -----------------------------------------------------------------------
# se3_inv
# -----------------------------------------------------------------------

class TestSe3Inv:
    def test_identity(self):
        I = torch.eye(4).unsqueeze(0)
        assert torch.allclose(se3_inv(I), I, atol=1e-6)

    def test_round_trip(self):
        mat = _random_se3(5)
        inv = se3_inv(mat)
        identity = mat @ inv
        expected = torch.eye(4).expand_as(identity)
        assert torch.allclose(identity, expected, atol=1e-5)

    def test_arbitrary_batch_dims(self):
        mat = _random_se3(6).reshape(2, 3, 4, 4)
        inv = se3_inv(mat)
        identity = mat @ inv
        expected = torch.eye(4).expand_as(identity)
        assert torch.allclose(identity, expected, atol=1e-5)

    def test_3x4_input(self):
        mat44 = _random_se3(1)
        mat34 = mat44[:, :3, :]
        inv = se3_inv(mat34)
        assert inv.shape == (1, 4, 4)
        identity = mat44 @ inv
        assert torch.allclose(identity, torch.eye(4).unsqueeze(0), atol=1e-5)


# -----------------------------------------------------------------------
# qvec_to_rotmat
# -----------------------------------------------------------------------

class TestQvecToRotmat:
    def test_identity_quaternion(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = qvec_to_rotmat(q)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)

    def test_orthogonality(self):
        q = torch.randn(10, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = qvec_to_rotmat(q)
        RtR = R.transpose(-1, -2) @ R
        assert torch.allclose(RtR, torch.eye(3).expand(10, 3, 3), atol=1e-5)

    def test_determinant_positive(self):
        q = torch.randn(10, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = qvec_to_rotmat(q)
        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(10), atol=1e-5)


# -----------------------------------------------------------------------
# CameraPose round-trip conversions
# -----------------------------------------------------------------------

class TestCameraPoseConversion:
    @pytest.mark.parametrize("src_conv", ALL_CONVENTIONS)
    @pytest.mark.parametrize("dst_conv", ALL_CONVENTIONS)
    def test_axis_round_trip(self, src_conv, dst_conv):
        """A -> B -> A should be identity for any pair."""
        mat = _random_se3(3)
        pose = CameraPose.from_matrix(mat, src_conv, ExtrinsicType.C2W)
        converted = pose.to(dst_conv).to(src_conv)
        assert torch.allclose(converted.matrix, mat, atol=1e-5), (
            f"{src_conv} -> {dst_conv} -> {src_conv} failed"
        )

    def test_c2w_w2c_round_trip(self):
        mat = _random_se3(3)
        pose = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.C2W)
        w2c = pose.to(extrinsic_type=ExtrinsicType.W2C)
        c2w = w2c.to(extrinsic_type=ExtrinsicType.C2W)
        assert torch.allclose(c2w.matrix, mat, atol=1e-5)

    def test_combined_round_trip(self):
        mat = _random_se3(3)
        pose = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.C2W)
        converted = pose.to(CameraConvention.BLENDER, ExtrinsicType.W2C)
        back = converted.to(CameraConvention.OPENCV, ExtrinsicType.C2W)
        assert torch.allclose(back.matrix, mat, atol=1e-5)

    def test_same_group_is_noop(self):
        mat = _random_se3(2)
        pose = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.C2W)
        colmap = pose.to(CameraConvention.COLMAP)
        assert torch.allclose(colmap.matrix, mat, atol=1e-7)

    def test_c2w_property(self):
        mat = _random_se3(2)
        pose_c2w = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.C2W)
        assert torch.allclose(pose_c2w.c2w, mat, atol=1e-6)
        assert torch.allclose(pose_c2w.w2c @ mat, torch.eye(4).expand(2, 4, 4), atol=1e-5)

    def test_w2c_property(self):
        mat = _random_se3(2)
        pose_w2c = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.W2C)
        assert torch.allclose(pose_w2c.w2c, mat, atol=1e-6)
        assert torch.allclose(pose_w2c.c2w @ mat, torch.eye(4).expand(2, 4, 4), atol=1e-5)

    def test_inverse_flips_type(self):
        mat = _random_se3(1)
        pose = CameraPose.from_matrix(mat, CameraConvention.OPENCV, ExtrinsicType.C2W)
        inv = pose.inverse()
        assert inv.extrinsic_type == ExtrinsicType.W2C
        assert inv.convention == CameraConvention.OPENCV
        identity = pose.matrix @ inv.matrix
        assert torch.allclose(identity, torch.eye(4).unsqueeze(0), atol=1e-5)


# -----------------------------------------------------------------------
# Factory methods
# -----------------------------------------------------------------------

class TestFactoryMethods:
    def test_from_Rt(self):
        R = qvec_to_rotmat(torch.tensor([[1.0, 0, 0, 0]]))
        t = torch.tensor([[1.0, 2.0, 3.0]])
        pose = CameraPose.from_Rt(R, t, CameraConvention.OPENCV, ExtrinsicType.C2W)
        assert pose.matrix.shape == (1, 4, 4)
        assert torch.allclose(pose.matrix[0, :3, :3], torch.eye(3), atol=1e-6)
        assert torch.allclose(pose.matrix[0, :3, 3], t[0], atol=1e-6)

    def test_from_qvec_tvec(self):
        qvec = torch.tensor([[1.0, 0, 0, 0]])
        tvec = torch.tensor([[0.0, 0, 0]])
        pose = CameraPose.from_qvec_tvec(qvec, tvec)
        assert pose.convention == CameraConvention.COLMAP
        assert pose.extrinsic_type == ExtrinsicType.W2C

    def test_from_pt3d(self):
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros(1, 3)
        pose = CameraPose.from_pt3d(R, T)
        assert pose.convention == CameraConvention.OPENCV
        assert pose.extrinsic_type == ExtrinsicType.C2W

    def test_from_matrix_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            CameraPose.from_matrix(torch.randn(3, 3), CameraConvention.OPENCV, ExtrinsicType.C2W)


# -----------------------------------------------------------------------
# Intrinsics
# -----------------------------------------------------------------------

class TestIntrinsics:
    def test_numpy_round_trip(self):
        K = np.array([[500, 0, 320.5], [0, 500, 240.5], [0, 0, 1]], dtype=np.float32)
        K_cv = colmap_to_opencv_intrinsics(K)
        K_back = opencv_to_colmap_intrinsics(K_cv)
        assert np.allclose(K, K_back)

    def test_torch_round_trip(self):
        K = torch.tensor([[500, 0, 320.5], [0, 500, 240.5], [0, 0, 1]], dtype=torch.float32)
        K_cv = colmap_to_opencv_intrinsics(K)
        K_back = opencv_to_colmap_intrinsics(K_cv)
        assert torch.allclose(K, K_back)

    def test_same_convention_noop(self):
        K = np.eye(3, dtype=np.float32)
        assert np.array_equal(
            convert_intrinsics(K, IntrinsicConvention.OPENCV, IntrinsicConvention.OPENCV), K
        )

    def test_delta_value(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        K_colmap = opencv_to_colmap_intrinsics(K)
        assert K_colmap[0, 2] == pytest.approx(320.5)
        assert K_colmap[1, 2] == pytest.approx(240.5)


# -----------------------------------------------------------------------
# PLY I/O
# -----------------------------------------------------------------------

class TestPlyIO:
    def test_round_trip(self):
        G = 50
        means = torch.randn(G, 3)
        scales = torch.rand(G, 3) * 0.1 + 0.01
        rotations = torch.randn(G, 4)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        harmonics = torch.randn(G, 3, 1)
        opacities = torch.rand(G)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.ply"
            export_gaussian_ply(
                means, scales, rotations, harmonics, opacities, path,
                convention=CameraConvention.OPENCV,
            )
            data = load_gaussian_ply(path)

        assert data.convention == CameraConvention.OPENCV
        assert data.means.shape == (G, 3)
        assert torch.allclose(data.means, means, atol=1e-4)
        assert torch.allclose(data.opacities, opacities, atol=1e-3)
        assert data.scales.shape == (G, 3)

    def test_convention_in_header(self):
        G = 10
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.ply"
            export_gaussian_ply(
                torch.randn(G, 3), torch.rand(G, 3), torch.randn(G, 4),
                torch.randn(G, 3, 1), torch.rand(G), path,
                convention=CameraConvention.BLENDER,
            )
            data = load_gaussian_ply(path)
            assert data.convention == CameraConvention.BLENDER


# -----------------------------------------------------------------------
# repr
# -----------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_info(self):
        pose = CameraPose.from_matrix(
            torch.eye(4).unsqueeze(0), CameraConvention.OPENCV, ExtrinsicType.C2W
        )
        r = repr(pose)
        assert "opencv" in r
        assert "c2w" in r
        assert "[1, 4, 4]" in r

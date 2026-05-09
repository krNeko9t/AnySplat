"""Tests for transforms JSON camera loading (GPU smoke optional)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.coord import CameraConvention, CameraPose, ExtrinsicType
from src.trace_cameras.load_transforms_json import (
    _axis_convention_from_string,
    load_transforms_json_cameras,
    resolve_transforms_json_path,
)


def test_axis_convention_from_string():
    assert _axis_convention_from_string("nerfstudio") == CameraConvention.NERFSTUDIO
    assert _axis_convention_from_string("blender") == CameraConvention.NERFSTUDIO
    assert _axis_convention_from_string("opencv") == CameraConvention.OPENCV


def test_resolve_explicit(tmp_path: Path):
    root = tmp_path / "scene"
    root.mkdir()
    explicit = root / "my.json"
    explicit.write_text("{}", encoding="utf-8")
    got = resolve_transforms_json_path(str(root), str(explicit))
    assert got == str(explicit.resolve())


def test_resolve_fallback_transforms_train(tmp_path: Path):
    root = tmp_path / "scene"
    root.mkdir()
    (root / "transforms_train.json").write_text('{"frames":[]}', encoding="utf-8")
    got = resolve_transforms_json_path(str(root), None)
    assert got.endswith("transforms_train.json")


def test_extrinsics_rt_matches_manual_pose_chain():
    """Golden: same decomposition as load_transforms_json_cameras inner loop."""
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[0, 3] = 0.25
    c2w[1, 3] = -0.5
    c2w[2, 3] = 1.75
    pose = CameraPose.from_matrix(
        c2w.unsqueeze(0), CameraConvention.NERFSTUDIO, ExtrinsicType.C2W
    )
    w2c = pose.to(CameraConvention.OPENCV, ExtrinsicType.W2C).matrix.squeeze(0)
    w2c_np = w2c.detach().cpu().numpy().astype(np.float32)
    R = w2c_np[:3, :3].T
    T = w2c_np[:3, 3]

    w2c_rt = np.eye(4, dtype=np.float32)
    w2c_rt[:3, :3] = R.T
    w2c_rt[:3, 3] = T
    torch.testing.assert_close(torch.from_numpy(w2c_rt), w2c.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TraceCamera needs CUDA")
def test_load_transforms_json_trace_camera_smoke(tmp_path: Path):
    scene = tmp_path / "scene"
    img_dir = scene / "images"
    img_dir.mkdir(parents=True)
    dummy_png = img_dir / "frame_00001.png"
    dummy_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    spec = {
        "camera_model": "OPENCV",
        "w": 640,
        "h": 480,
        "fl_x": 500.0,
        "fl_y": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "frames": [
            {
                "file_path": "images/frame_00001.png",
                "transform_matrix": np.eye(4, dtype=float).tolist(),
            },
        ],
    }
    json_path = scene / "transforms_train.json"
    json_path.write_text(json.dumps(spec), encoding="utf-8")

    cams = load_transforms_json_cameras(
        str(json_path), resolution=1, transforms_axis="nerfstudio"
    )
    assert len(cams) == 1
    assert cams[0].image_width == 640
    assert cams[0].image_height == 480

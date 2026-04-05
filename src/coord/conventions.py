"""
Coordinate system conventions used in 3D vision and graphics.

All conventions define a RIGHT-HANDED coordinate system.
They differ only in which axis points where in camera space.

=== Camera Axis Conventions ===

There are two groups of camera-axis conventions:

  "opencv-like" (OpenCV, COLMAP):

        +Z (forward / looking direction)
       /
      /
     o-------> +X (right)
     |
     |
     v +Y (down)

  "opengl-like" (OpenGL, Blender, NerfStudio):

     ^ +Y (up)
     |
     |
     o-------> +X (right)
    /
   /
  +Z (backward / behind the camera)

  Conversion between the two groups: right-multiply c2w (or left-multiply
  w2c) by diag(1, -1, -1, 1).  The matrix is its own inverse.

=== Intrinsic Pixel Center ===

  OpenCV:  top-left pixel center at (0, 0)
  COLMAP:  top-left pixel center at (0.5, 0.5)
  Difference is just ±0.5 on cx, cy.
"""

from enum import Enum

import torch
from torch import Tensor


class CameraConvention(str, Enum):
    """Camera coordinate axis convention."""
    OPENCV = "opencv"
    COLMAP = "colmap"
    OPENGL = "opengl"
    BLENDER = "blender"
    NERFSTUDIO = "nerfstudio"


class ExtrinsicType(str, Enum):
    """Whether the 4x4 matrix maps camera->world or world->camera."""
    C2W = "c2w"
    W2C = "w2c"


class IntrinsicConvention(str, Enum):
    """Pixel-center convention for the intrinsic matrix."""
    OPENCV = "opencv"   # top-left pixel center at (0, 0)
    COLMAP = "colmap"   # top-left pixel center at (0.5, 0.5)


# ---------------------------------------------------------------------------
# Axis-group classification
# ---------------------------------------------------------------------------

AXIS_GROUP: dict[CameraConvention, str] = {
    CameraConvention.OPENCV: "opencv",
    CameraConvention.COLMAP: "opencv",
    CameraConvention.OPENGL: "opengl",
    CameraConvention.BLENDER: "opengl",
    CameraConvention.NERFSTUDIO: "opengl",
}

# The single matrix that flips between the two axis groups.
# It is its own inverse: FLIP @ FLIP == I.
AXIS_FLIP: Tensor = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

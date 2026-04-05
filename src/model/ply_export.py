"""Legacy PLY export — delegates to ``src.coord.ply_io.export_gaussian_ply``.

New code should use ``src.coord.export_gaussian_ply`` directly.
"""

from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from src.coord import CameraConvention, export_gaussian_ply


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    opacity_in_prob_space: bool = True,
):
    """Write Gaussians to a standard 3DGS PLY (OpenCV coordinate convention).

    Thin wrapper around ``src.coord.export_gaussian_ply`` that preserves the
    original call signature.
    """
    export_gaussian_ply(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,
        opacities=opacities,
        path=path,
        convention=CameraConvention.OPENCV,
        scales_in_log=False,
        opacity_in_logit=not opacity_in_prob_space,
        save_sh_dc_only=save_sh_dc_only,
        shift_and_scale=shift_and_scale,
    )

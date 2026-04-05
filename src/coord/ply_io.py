"""
Unified 3D Gaussian Splatting PLY import / export.

Standard 3DGS PLY on-disk format
---------------------------------
- positions:   ``x, y, z``  (world coordinates)
- normals:     ``nx, ny, nz``  (zeros)
- SH DC:       ``f_dc_0, f_dc_1, f_dc_2``
- SH rest:     ``f_rest_0 .. f_rest_N``  (optional)
- opacity:     ``opacity``  — stored in **logit** (inverse-sigmoid) space
- scales:      ``scale_0, scale_1, scale_2``  — stored in **log** space
- quaternions: ``rot_0, rot_1, rot_2, rot_3``  — **(w, x, y, z)** order

This module always converts to/from the "friendly" representation used at
runtime (linear scales, probability opacity) so callers don't need to worry
about log/logit encoding.

A ``comment coordinate_convention=<value>`` line is written into the PLY
header so readers can identify the coordinate system without guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .conventions import CameraConvention

logger = logging.getLogger("coord")

# SH normalisation constant  1 / (2 * sqrt(pi))
_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GaussianPlyData:
    """Runtime-friendly Gaussian parameters loaded from a PLY file."""
    means: Tensor           # (G, 3) world-space positions
    scales: Tensor          # (G, 2|3) **linear** scale (not log)
    rotations: Tensor       # (G, 4) quaternion (w, x, y, z), normalised
    opacities: Tensor       # (G,)   probability in [0, 1] (not logit)
    sh_dc: Tensor           # (G, 3) SH DC band
    sh_rest: Tensor         # (G, N) higher SH bands (may be empty)
    colors_rgb: Tensor      # (G, 3) approximate RGB from SH DC, in [0, 1]
    convention: Optional[CameraConvention]  # read from header, or None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _build_attributes(num_sh_rest: int) -> list[str]:
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attrs.append(f"f_dc_{i}")
    for i in range(num_sh_rest):
        attrs.append(f"f_rest_{i}")
    attrs.append("opacity")
    for i in range(3):
        attrs.append(f"scale_{i}")
    for i in range(4):
        attrs.append(f"rot_{i}")
    return attrs


def export_gaussian_ply(
    means: Tensor,
    scales: Tensor,
    rotations: Tensor,
    harmonics: Tensor,
    opacities: Tensor,
    path: Path | str,
    *,
    convention: CameraConvention = CameraConvention.OPENCV,
    scales_in_log: bool = False,
    opacity_in_logit: bool = False,
    save_sh_dc_only: bool = True,
    shift_and_scale: bool = False,
) -> None:
    """Write Gaussians to a standard 3DGS PLY file.

    Parameters
    ----------
    means : (G, 3) world positions.
    scales : (G, 3) gaussian scales in **linear** space (unless *scales_in_log*).
    rotations : (G, 4) quaternion (w,x,y,z).
    harmonics : (G, 3, D_sh) spherical harmonics.
    opacities : (G,) in [0,1] probability space (unless *opacity_in_logit*).
    path : output file.
    convention : which coordinate system *means* are expressed in.
    scales_in_log : set True if *scales* are already in log-space.
    opacity_in_logit : set True if *opacities* are already in logit-space.
    save_sh_dc_only : drop higher-order SH bands to save space.
    shift_and_scale : centre and normalise the scene to [-1, 1].
    """
    from plyfile import PlyData, PlyElement

    path = Path(path)

    if shift_and_scale:
        means = means - means.median(dim=0).values
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        if not scales_in_log:
            scales = scales / scale_factor

    # --- to numpy ---------------------------------------------------------
    means_np = means.detach().cpu().float().numpy()
    normals_np = np.zeros_like(means_np)

    # quaternions: normalise, keep (w,x,y,z)
    rot_np = rotations.detach().cpu().float().numpy()
    rot_np = rot_np / (np.linalg.norm(rot_np, axis=1, keepdims=True) + 1e-8)

    # SH
    f_dc = harmonics[..., 0].detach().cpu().contiguous().float().numpy()
    f_rest = harmonics[..., 1:].flatten(start_dim=1).detach().cpu().contiguous().float().numpy()

    # opacity -> logit
    if opacity_in_logit:
        opacity_np = opacities.detach().cpu().float().numpy()
    else:
        op_clamped = opacities.detach().float().clamp(1e-6, 1.0 - 1e-6)
        opacity_np = torch.logit(op_clamped).cpu().numpy()
    opacity_np = opacity_np.reshape(-1, 1)

    # scales -> log
    if scales_in_log:
        scales_np = scales.detach().cpu().float().numpy()
    else:
        scales_np = scales.detach().cpu().float().clamp(min=1e-8).log().numpy()

    # --- assemble structured array ----------------------------------------
    num_rest = 0 if save_sh_dc_only else f_rest.shape[1]
    attr_names = _build_attributes(num_rest)
    dtype_full = [(a, "f4") for a in attr_names]

    parts = [means_np, normals_np, f_dc]
    if not save_sh_dc_only:
        parts.append(f_rest)
    parts.extend([opacity_np, scales_np, rot_np])

    data = np.concatenate(parts, axis=1)
    elements = np.empty(means_np.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, data))

    # --- write with convention comment ------------------------------------
    path.parent.mkdir(exist_ok=True, parents=True)
    ply = PlyData([PlyElement.describe(elements, "vertex")],
                  comments=[f"coordinate_convention={convention.value}"])
    ply.write(str(path))

    logger.debug(
        "[COORD] export_gaussian_ply | %s | G=%d | %s",
        convention.value, means_np.shape[0], path,
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def load_gaussian_ply(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> GaussianPlyData:
    """Read a 3DGS PLY and return runtime-friendly (linear/prob) parameters.

    The coordinate convention is read from the PLY header comment.  If absent,
    a warning is logged and ``convention`` will be ``None``.
    """
    from plyfile import PlyData

    path = Path(path)
    plydata = PlyData.read(str(path))
    v = plydata.elements[0]

    # --- convention from header -------------------------------------------
    convention = None
    for c in plydata.comments:
        if c.startswith("coordinate_convention="):
            val = c.split("=", 1)[1].strip()
            try:
                convention = CameraConvention(val)
            except ValueError:
                logger.warning("[COORD] Unknown convention '%s' in %s", val, path)
    if convention is None:
        logger.warning(
            "[COORD] No coordinate_convention comment in %s — "
            "coordinate system is unknown!", path,
        )

    # --- positions --------------------------------------------------------
    means = torch.tensor(
        np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1),
        dtype=torch.float32, device=device,
    )

    # --- scales (log -> linear) -------------------------------------------
    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales_log = np.stack([np.asarray(v[n]) for n in scale_names], axis=1)
    scales = torch.tensor(scales_log, dtype=torch.float32, device=device).exp()

    # --- opacity (logit -> prob) ------------------------------------------
    opacities = torch.tensor(
        np.asarray(v["opacity"]), dtype=torch.float32, device=device,
    ).sigmoid()

    # --- quaternions (w, x, y, z) -----------------------------------------
    rot = torch.tensor(
        np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], axis=1),
        dtype=torch.float32, device=device,
    )
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)

    # --- SH ---------------------------------------------------------------
    sh_dc = torch.tensor(
        np.stack([np.asarray(v[f"f_dc_{i}"]) for i in range(3)], axis=1),
        dtype=torch.float32, device=device,
    )
    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        sh_rest = torch.tensor(
            np.stack([np.asarray(v[n]) for n in rest_names], axis=1),
            dtype=torch.float32, device=device,
        )
    else:
        sh_rest = torch.empty(means.shape[0], 0, dtype=torch.float32, device=device)

    # --- approximate RGB --------------------------------------------------
    colors_rgb = (0.5 + _C0 * sh_dc).clamp(0.0, 1.0)

    logger.debug(
        "[COORD] load_gaussian_ply | %s | G=%d, scales=%d | %s",
        convention.value if convention else "UNKNOWN",
        means.shape[0], scales.shape[1], path,
    )

    return GaussianPlyData(
        means=means,
        scales=scales,
        rotations=rot,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        colors_rgb=colors_rgb,
        convention=convention,
    )

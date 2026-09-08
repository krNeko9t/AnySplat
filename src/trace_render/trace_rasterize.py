"""Load Gaussians and run CUDA ``trace()`` for RGB (no CLI)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from src.coord import (
    CameraConvention,
    ExtrinsicType,
    load_gaussian_ply,
    log_coordinate_op,
)

# Must match TRACE_CHANNELS in diff_surfel_rasterization and
# diff_gaussian_rasterization cuda_rasterizer/config.h
TRACE_CHANNELS = 20


def resolve_trace_backend(requested: str, n_scales: int) -> str:
    """Return ``surfel`` or ``3dgs`` from *requested* and PLY scale count."""
    if requested == "auto":
        if n_scales == 2:
            return "surfel"
        if n_scales == 3:
            return "3dgs"
        raise ValueError(
            "trace_backend=auto requires PLY with 2 (2DGS) or 3 (3DGS) scale "
            f"attributes; this file has {n_scales}. Set trace_backend explicitly."
        )
    if requested == "surfel" and n_scales != 2:
        raise ValueError(
            "trace_backend=surfel expects 2DGS PLY (2 scale_*) entries; "
            f"got {n_scales}."
        )
    if requested == "3dgs" and n_scales != 3:
        raise ValueError(
            "trace_backend=3dgs expects 3DGS PLY (3 scale_*) entries; "
            f"got {n_scales}."
        )
    return requested


def effective_trace_backend(
    requested: str,
    n_scales: int,
    ckpt_backend: str | None,
) -> str:
    """Resolve backend for checkpoints: explicit *requested* overrides *ckpt_backend*."""
    if requested != "auto":
        return resolve_trace_backend(requested, n_scales)
    if ckpt_backend in ("surfel", "3dgs"):
        return ckpt_backend
    return resolve_trace_backend("auto", n_scales)


def load_gaussians_from_ply(ply_path: str | Path):
    """Load Gaussian parameters for trace via :func:`src.coord.load_gaussian_ply`."""
    path_s = os.fspath(ply_path)
    print(f"Loading PLY: {path_s}")
    data = load_gaussian_ply(path_s, device="cuda")
    means = data.means
    quats = data.rotations
    scales_linear = data.scales
    opacities = data.opacities.unsqueeze(-1)
    colors = data.colors_rgb
    conv = data.convention if data.convention is not None else CameraConvention.OPENCV
    if data.convention is not None:
        print(f"  coordinate_convention (header): {data.convention.value}")
    log_coordinate_op(
        "load_gaussian_ply",
        conv,
        ExtrinsicType.C2W,
        tuple(means.shape),
        context="Gaussian means (world); PLY header convention",
    )
    n_scales = scales_linear.shape[1]
    print(f"  Gaussians: {means.shape[0]:,}, scales: {n_scales}D (linear space)")
    return means, quats, scales_linear, opacities, colors


def _trace_single_view_surfel(
    means, quats, scales, opacities, colors,
    img_sem, img_mask, cam, bg_color,
):
    try:
        from diff_surfel_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as e:
        raise RuntimeError(
            "trace_backend=surfel requires diff_surfel_rasterization. "
            "Original error: " + str(e)
        ) from e

    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=float(cam.tanfovx),
        tanfovy=float(cam.tanfovy),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=0,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(means, requires_grad=False)

    with torch.no_grad():
        out_color, _gau_depth, gau_sem, num_ray, radii = rasterizer.trace(
            means3D=means,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            img_sem=img_sem,
            img_mask=img_mask,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )

    return gau_sem, num_ray, radii, out_color


def _trace_single_view_3dgs(
    means, quats, scales, opacities, colors,
    img_sem, img_mask, cam, bg_color,
):
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as e:
        raise RuntimeError(
            "trace_backend=3dgs requires diff_gaussian_rasterization "
            "(install the CUDA extension). Original error: " + str(e)
        ) from e

    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=float(cam.tanfovx),
        tanfovy=float(cam.tanfovy),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=0,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(means, requires_grad=False)

    with torch.no_grad():
        out_color, _gau_depth, gau_sem, num_gsem, radii = rasterizer.trace(
            means3D=means,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            img_sem=img_sem,
            img_mask=img_mask,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )

    return gau_sem, num_gsem, radii, out_color


def trace_single_view(
    means, quats, scales, opacities, colors,
    img_sem, img_mask, cam, bg_color,
    trace_backend: str,
):
    """Run rasterizer ``trace`` for one camera view."""
    if trace_backend == "surfel":
        return _trace_single_view_surfel(
            means, quats, scales, opacities, colors,
            img_sem, img_mask, cam, bg_color,
        )
    if trace_backend == "3dgs":
        return _trace_single_view_3dgs(
            means, quats, scales, opacities, colors,
            img_sem, img_mask, cam, bg_color,
        )
    raise ValueError(f"Unknown trace_backend: {trace_backend!r}")


def raster_gaussian_rgb_u8(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cam,
    *,
    trace_backend: str,
    bg_color_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """RGB-only Gaussian splat for one view → uint8 H×W×3."""
    device = means.device
    h = int(cam.image_height)
    w = int(cam.image_width)
    img_sem = torch.zeros(h, w, TRACE_CHANNELS, device=device, dtype=torch.float32)
    img_mask = torch.ones(h, w, dtype=torch.int32, device=device)
    bg = torch.tensor(bg_color_rgb, dtype=torch.float32, device=device)
    with torch.no_grad():
        _a, _b, _radii, out_color = trace_single_view(
            means,
            quats,
            scales,
            opacities,
            colors,
            img_sem,
            img_mask,
            cam,
            bg,
            trace_backend,
        )
    return (
        out_color.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)


def trace_single_view_chunked(
    means, quats, scales, opacities, colors,
    feat_hwc, img_mask, cam, bg_color,
    trace_backend: str,
):
    """Trace a feature map of arbitrary depth by splitting it into TRACE_CHANNELS passes.

    ``feat_hwc`` is ``[H, W, D]`` with *any* ``D >= 1``; the CUDA kernels only
    accept exactly ``TRACE_CHANNELS`` channels, so the map is cut into
    ``ceil(D / TRACE_CHANNELS)`` slices (the last one zero-padded) and the
    per-Gaussian results concatenated back along the channel axis.

    This is exact, not an approximation: inside ``trace`` the channels never mix
    (``gau_sem[:, c]`` is an alpha-weighted accumulation of ``img_sem[:, :, c]``
    alone), so the concatenation is element-wise equal to one hypothetical
    D-channel trace.

    ``num_ray``, ``radii`` and ``out_color`` depend only on the geometry and on
    ``img_mask``, so every pass returns the same values; they are taken from the
    first pass. Summing them instead would inflate the caller's normalization
    denominator by the number of passes.

    Returns ``(gau_sem [P, D], num_ray, radii, out_color)``.
    """
    if feat_hwc.dim() != 3:
        raise ValueError(f"feat_hwc must be [H, W, D]; got shape {tuple(feat_hwc.shape)}")
    h, w, feat_dim = feat_hwc.shape
    if feat_dim < 1:
        raise ValueError(f"feat_hwc must have at least 1 channel; got {feat_dim}")

    gau_sem_parts = []
    shared = None

    for start in range(0, feat_dim, TRACE_CHANNELS):
        stop = min(start + TRACE_CHANNELS, feat_dim)
        chunk = feat_hwc[:, :, start:stop]
        if stop - start < TRACE_CHANNELS:
            pad = torch.zeros(
                h, w, TRACE_CHANNELS - (stop - start),
                device=feat_hwc.device, dtype=feat_hwc.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=2)
        gau_sem, num_ray, radii, out_color = trace_single_view(
            means, quats, scales, opacities, colors,
            chunk.contiguous(), img_mask, cam, bg_color,
            trace_backend,
        )
        gau_sem_parts.append(gau_sem[:, : stop - start])
        if shared is None:
            shared = (num_ray, radii, out_color)

    return torch.cat(gau_sem_parts, dim=1), *shared

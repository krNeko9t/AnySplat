from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as F
from jaxtyping import Float, Int64
from torch import Tensor


@dataclass(frozen=True)
class ResizeCropParams:
    # After resize-to-cover (scale_factor) and center-crop to (h_out, w_out).
    scale_factor: float
    h_scaled: int
    w_scaled: int
    top: int
    left: int
    h_out: int
    w_out: int


def resize_to_cover_and_center_crop_params(
    h_in: int,
    w_in: int,
    h_out: int,
    w_out: int,
) -> ResizeCropParams:
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = int(round(h_in * scale_factor))
    w_scaled = int(round(w_in * scale_factor))
    top = (h_scaled - h_out) // 2
    left = (w_scaled - w_out) // 2
    return ResizeCropParams(
        scale_factor=scale_factor,
        h_scaled=h_scaled,
        w_scaled=w_scaled,
        top=top,
        left=left,
        h_out=h_out,
        w_out=w_out,
    )


def apply_resize_crop_rgb(
    img: Float[Tensor, "3 h w"],
    p: ResizeCropParams,
) -> Float[Tensor, "3 h_out w_out"]:
    img = F.resize(img, [p.h_scaled, p.w_scaled], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    img = img[:, p.top : p.top + p.h_out, p.left : p.left + p.w_out]
    return img


def apply_resize_crop_depth(
    depth: Float[Tensor, "h w"],
    p: ResizeCropParams,
) -> Float[Tensor, "h_out w_out"]:
    # Keep as 1xHxW for torchvision resize.
    d = depth[None]
    d = F.resize(d, [p.h_scaled, p.w_scaled], interpolation=F.InterpolationMode.BILINEAR, antialias=False)
    d = d[:, p.top : p.top + p.h_out, p.left : p.left + p.w_out]
    return d[0]


def apply_resize_crop_instance_mask(
    mask: Int64[Tensor, "h w"],
    p: ResizeCropParams,
) -> Int64[Tensor, "h_out w_out"]:
    m = mask[None].to(torch.float32)
    m = F.resize(m, [p.h_scaled, p.w_scaled], interpolation=F.InterpolationMode.NEAREST)
    m = m[:, p.top : p.top + p.h_out, p.left : p.left + p.w_out]
    return m[0].to(torch.int64)


def adjust_K_for_resize_crop_and_normalize(
    K: Float[Tensor, "3 3"],
    h_in: int,
    w_in: int,
    p: ResizeCropParams,
) -> Float[Tensor, "3 3"]:
    """Convert pixel-space K to normalized K after resize+crop.

    Assumes K is in pixel units for the original image of size (h_in, w_in).
    Output K is normalized to (h_out, w_out): fx,cx divided by w_out; fy,cy divided by h_out.
    """
    K2 = K.clone()
    s = p.scale_factor
    # Scale.
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    # Center-crop offset.
    K2[0, 2] -= float(p.left)
    K2[1, 2] -= float(p.top)
    # Normalize.
    K2n = K2.clone()
    K2n[0, :] /= float(p.w_out)
    K2n[1, :] /= float(p.h_out)
    return K2n


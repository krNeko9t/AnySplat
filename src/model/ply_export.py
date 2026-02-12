from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


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
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Model outputs quaternions in (w, x, y, z) scalar-first convention,
    # which is also what the standard 3DGS PLY format expects (rot_0=w, ...).
    # Just normalize and write directly — no scipy conversion needed.
    rot_np = rotations.detach().cpu().float().numpy()
    rot_np = rot_np / (np.linalg.norm(rot_np, axis=1, keepdims=True) + 1e-8)
    rotations = rot_np

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    # Standard 3DGS PLY stores opacity in inverse-sigmoid (logit) space.
    # The model's opacities are in probability space [0,1] after map_pdf_to_opacity.
    # Convert to logit so viewers (which apply sigmoid on load) recover the
    # correct opacity.
    if opacity_in_prob_space:
        op_clamped = opacities.detach().float().clamp(1e-6, 1.0 - 1e-6)
        opacity_for_ply = torch.logit(op_clamped)[..., None].cpu().numpy()
    else:
        opacity_for_ply = opacities[..., None].detach().cpu().numpy()

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacity_for_ply,
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from src.model.types import Gaussians


def export_gaussian_instance_embedding(
    output_path: Path,
    gaussians: Gaussians,
    gaussian_instance_feat: Float[Tensor, "batch gaussian n"],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save gs-wise instance embedding aligned with Gaussian order.

    The saved file is a torch checkpoint containing:
    - means: [G,3]
    - instance_feat: [G,N]
    - meta: dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "means": gaussians.means[0].detach().cpu(),
        "instance_feat": gaussian_instance_feat[0].detach().cpu(),
        "meta": meta or {},
    }
    torch.save(payload, output_path)
    return output_path


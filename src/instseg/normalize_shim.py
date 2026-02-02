from __future__ import annotations

import torch
from torch import Tensor

from .types import BatchedExample


def normalize_image(tensor: Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> Tensor:
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1, 1, 1)
    return (tensor - mean_t) / std_t


def apply_normalize_shim(
    batch: BatchedExample,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> BatchedExample:
    # Both context and target are normalized to [-1, 1] to keep downstream assumptions consistent.
    if "context" in batch and "image" in batch["context"]:
        batch["context"]["image"] = normalize_image(batch["context"]["image"], mean, std)
    if "target" in batch and "image" in batch["target"]:
        batch["target"]["image"] = normalize_image(batch["target"]["image"], mean, std)
    return batch


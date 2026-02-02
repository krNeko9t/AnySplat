from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def kmeans_torch(
    x: Tensor,
    k: int,
    num_iters: int = 30,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Simple k-means on GPU/CPU.

    Args:
        x: [N, D] float tensor.
        k: number of clusters.
    Returns:
        labels: [N] long
        centers: [k, D]
    """
    if x.ndim != 2:
        raise ValueError("x must be [N,D]")
    N, D = x.shape
    if k <= 0 or k > N:
        raise ValueError("k must be in [1, N]")

    # Normalize embeddings (often beneficial for contrastive embeddings).
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)

    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    # Random init.
    init_idx = torch.randperm(N, generator=g, device=x.device)[:k]
    centers = x[init_idx].clone()

    for _ in range(num_iters):
        # Assign.
        # Using cosine distance since x is normalized: d^2 = 2-2dot.
        dot = x @ centers.T  # [N,k]
        dist2 = 2.0 - 2.0 * dot
        labels = dist2.argmin(dim=1)  # [N]

        # Update.
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros((k,), device=x.device, dtype=torch.int64)
        new_centers.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.int64))
        # Avoid empty clusters by keeping old center.
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        new_centers[~mask] = centers[~mask]
        centers = F.normalize(new_centers, p=2, dim=-1, eps=1e-8)

    # Final labels.
    dot = x @ centers.T
    labels = (2.0 - 2.0 * dot).argmin(dim=1)
    return labels.to(torch.int64), centers


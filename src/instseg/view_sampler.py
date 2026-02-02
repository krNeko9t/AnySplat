from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int = 0


ViewSamplerCfg = ViewSamplerBoundedCfg


def sample_bounded(
    num_views: int,
    cfg: ViewSamplerBoundedCfg,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[
    Int64[Tensor, "context_view"],
    Int64[Tensor, "target_view"],
]:
    """A simplified bounded sampler (compatible with AnySplat training needs)."""
    if num_views < 2:
        raise ValueError("Need at least 2 views to sample context.")

    # Gap between left/right context.
    min_gap = max(cfg.min_distance_between_context_views, 1)
    max_gap = min(cfg.max_distance_between_context_views, num_views - 1)
    if max_gap < min_gap:
        max_gap = min_gap

    gap = torch.randint(min_gap, max_gap + 1, size=tuple(), device=device, generator=generator).item()

    left = torch.randint(0, num_views - gap, size=tuple(), device=device, generator=generator).item()
    right = left + gap

    # If more than 2 context views desired, sample extra views in-between.
    if cfg.num_context_views < 2:
        raise ValueError("num_context_views must be >= 2")
    if cfg.num_context_views == 2:
        context = torch.tensor([left, right], dtype=torch.int64, device=device)
    else:
        extra = []
        num_extra = cfg.num_context_views - 2
        while len(set(extra)) != num_extra:
            extra = torch.randint(left + 1, right, (num_extra,), device=device, generator=generator).tolist()
        context = torch.tensor([left, *extra, right], dtype=torch.int64, device=device)

    # Targets are sampled inside the interval, optionally away from the context endpoints.
    t_low = left + cfg.min_distance_to_context_views
    t_high = right - cfg.min_distance_to_context_views
    if t_high < t_low:
        t_low, t_high = left, right
    if cfg.num_target_views <= 0:
        target = torch.empty((0,), dtype=torch.int64, device=device)
    else:
        target = torch.randint(
            t_low,
            t_high + 1,
            (cfg.num_target_views,),
            device=device,
            generator=generator,
        )
    return context, target


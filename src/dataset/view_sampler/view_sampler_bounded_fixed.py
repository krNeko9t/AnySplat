from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedFixedCfg:
    """
    A drop-in replacement for `ViewSamplerBoundedCfg` that avoids the infinite loop when
    `num_context_views` requests more unique "middle" frames than are available between
    the two endpoints.

    Behavior when there are not enough middle frames:
    - Select as many unique middle frames as possible (without replacement).
    - If still short, fill the remainder by sampling with replacement (duplicates allowed).
    This keeps the output length stable (always returns `num_context_views` context indices).
    """

    name: Literal["bounded_fixed"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    max_img_per_gpu: int
    min_gap_multiplier: int
    max_gap_multiplier: int


class ViewSamplerBoundedFixed(ViewSampler[ViewSamplerBoundedFixedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        num_context_views: int,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        else:
            min_gap, max_gap = self.num_ctxt_gap_mapping[num_context_views]

        max_gap = min(max_gap, num_views - 1)
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")

        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        else:
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones. Original implementation used a while loop that can
        # become infinite when the requested number of unique middle views is impossible.
        extra_views_list: list[int]
        if num_context_views > 2:
            num_extra_views = num_context_views - 2

            # Candidate middle indices are in the open interval (left, right).
            # Note: For circular datasets this interval is ambiguous; we keep the
            # behavior consistent with the original sampler (sample before modulo).
            if index_context_right > index_context_left + 1:
                candidates = torch.arange(
                    index_context_left + 1,
                    index_context_right,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                candidates = torch.empty((0,), dtype=torch.int64, device=device)

            n_avail = int(candidates.numel())
            k_unique = min(num_extra_views, n_avail)

            if k_unique > 0:
                perm = torch.randperm(n_avail, device=device)[:k_unique]
                extra_unique = candidates[perm]
            else:
                extra_unique = torch.empty((0,), dtype=torch.int64, device=device)

            need = num_extra_views - int(extra_unique.numel())
            if need > 0:
                # Not enough unique middle frames; fill the remainder with replacement.
                # Prefer sampling from available middle candidates; if none exist,
                # fall back to repeating the left endpoint (degenerate but finite).
                if n_avail > 0:
                    fill_idx = torch.randint(0, n_avail, (need,), device=device)
                    extra_fill = candidates[fill_idx]
                else:
                    extra_fill = torch.full((need,), int(index_context_left), dtype=torch.int64, device=device)
                extra_all = torch.cat([extra_unique, extra_fill], dim=0)
            else:
                extra_all = extra_unique

            extra_views_list = extra_all.to(torch.int64).tolist()
        else:
            extra_views_list = []

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy
        return (
            torch.tensor((index_context_left, *extra_views_list, index_context_right), dtype=torch.int64, device=device),
            index_target.to(torch.int64),
            overlap,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

    @property
    def num_ctxt_gap_mapping(self) -> dict:
        mapping = dict()
        for num_ctxt in range(2, self.cfg.num_context_views + 1):
            mapping[num_ctxt] = [
                min(num_ctxt * self.cfg.min_gap_multiplier, self.cfg.min_distance_between_context_views),
                min(
                    max(num_ctxt * self.cfg.max_gap_multiplier, num_ctxt**2),
                    self.cfg.max_distance_between_context_views,
                ),
            ]
        return mapping


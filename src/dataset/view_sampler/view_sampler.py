import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    is_overfitting: bool
    cameras_are_circular: bool
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

    @abstractmethod
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()

    def scene_generator(
        self,
        scene: str,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Generator | None:
        """Per-scene RNG for the non-training stages; ``None`` (global RNG) for train.

        Validation frames used to come off the global RNG, which the training
        loop advances between validations, so every validation drew different
        frames and no cross-step curve was readable -- ``fixed_views_and_shape``
        pins only the view *count* and the resolution, not *which* frames
        (ticket 13.1).  Seeding off the scene id alone makes the draw a pure
        function of the scene: the same frames at every step, on every rank, in
        every dataloader worker, and in any later offline re-evaluation of a
        checkpoint.  Training is deliberately left on the global RNG -- its
        jitter is wanted.

        Python's ``hash`` is salted per process, so the digest is explicit.
        """
        if self.stage == "train":
            return None
        digest = hashlib.sha256(scene.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

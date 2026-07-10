"""Batch collation that preserves PhysicsTarget objects as a list.

default_collate cannot stack dataclasses or variable-length LUTs.
Physics targets stay as ``list[PhysicsTarget]`` (length = batch size).
"""

from __future__ import annotations

from torch.utils.data._utils.collate import default_collate

from src.dataset.physics.types import PhysicsTarget


def collate_examples(batch: list[dict]) -> dict:
    """Collate dataset examples; keep ``physics_target`` as a Python list."""
    targets = [ex.pop("physics_target", None) for ex in batch]
    collated = default_collate(batch)
    if any(t is not None for t in targets):
        # Replace None with a sentinel empty target? Prefer require all-or-nothing.
        if not all(t is not None for t in targets):
            raise ValueError(
                "physics_target must be present for every sample in the batch "
                "(or for none). Mixed batches are not supported."
            )
        collated["physics_target"] = targets  # list[PhysicsTarget]
    return collated

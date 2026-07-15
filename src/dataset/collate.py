"""Batch collation that preserves physics *Target objects as lists.

default_collate cannot stack dataclasses or variable-length LUTs.
Physics targets stay as Python lists (length = batch size).
"""

from __future__ import annotations

from torch.utils.data._utils.collate import default_collate

_PHYSICS_TARGET_KEYS = ("physics_target", "physics_property_target", "physgm_target")


def collate_examples(batch: list[dict]) -> dict:
    """Collate dataset examples; keep physics *Target fields as Python lists."""
    popped: dict[str, list] = {
        key: [ex.pop(key, None) for ex in batch] for key in _PHYSICS_TARGET_KEYS
    }
    collated = default_collate(batch)
    for key, values in popped.items():
        if not any(t is not None for t in values):
            continue
        if not all(t is not None for t in values):
            raise ValueError(
                f"{key} must be present for every sample in the batch "
                "(or for none). Mixed batches are not supported."
            )
        collated[key] = values
    return collated

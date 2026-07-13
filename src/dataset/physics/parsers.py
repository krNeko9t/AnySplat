"""Physics annotation parsers.

Add a new external format here:
  1. Implement ``PhysicsParser.parse(scene, root) -> PhysicsTarget | None``
  2. Register it in ``PHYSICS_PARSERS``
  3. Point ``dataset.manifest.physics_parser`` at the registry key
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import torch

from .types import PhysicsTarget

logger = logging.getLogger(__name__)


class PhysicsParser(Protocol):
    def parse(self, scene: dict, root: Path) -> PhysicsTarget | None:
        """Parse scene-level physics labels. Return None if absent."""
        ...


# 1-indexed class ids (0 reserved for ignore). Order defines CE class indices.
_3DOVS_CLASS_NAMES: tuple[str, ...] = ("static", "rigid", "soft", "unknown")
_3DOVS_LABEL_TO_ID: dict[str, int] = {
    name: i + 1 for i, name in enumerate(_3DOVS_CLASS_NAMES)
}


class ThreeDOVSJsonParser:
    """Parse ``physics_labels.json``: ``{instance_id_str: class_name, ...}``.

    Manifest field: ``physics_labels_path`` (relative to dataset root or absolute).
    """

    def parse(self, scene: dict, root: Path) -> PhysicsTarget | None:
        rel = scene.get("physics_labels_path")
        if rel is None:
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            logger.warning("[ThreeDOVSJsonParser] missing %s, skipping", path)
            return None
        with path.open("r") as f:
            raw: dict[str, str] = json.load(f)
        if not raw:
            return None

        id_to_cls = {
            int(k): _3DOVS_LABEL_TO_ID.get(v, 0) for k, v in raw.items()
        }
        max_id = max(id_to_cls.keys())
        lut = torch.zeros(max_id + 1, dtype=torch.int64)
        for inst_id, cls_int in id_to_cls.items():
            lut[inst_id] = cls_int
        return PhysicsTarget(label_lut=lut, class_names=_3DOVS_CLASS_NAMES)


PHYSICS_PARSERS: dict[str, PhysicsParser] = {
    "3dovs_json": ThreeDOVSJsonParser(),
}


def get_physics_parser(name: str | None) -> PhysicsParser | None:
    if name is None:
        return None
    if name not in PHYSICS_PARSERS:
        raise KeyError(
            f"Unknown physics_parser={name!r}. "
            f"Registered: {sorted(PHYSICS_PARSERS)}"
        )
    return PHYSICS_PARSERS[name]

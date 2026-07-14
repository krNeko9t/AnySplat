"""Physics annotation parsers.

Add a new external format here:
  1. Implement a parser with ``target_key`` + ``parse(scene, root)``
  2. Register it in ``PHYSICS_PARSERS``
  3. Point ``dataset.manifest.physics_parser`` at the registry key
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Protocol

import torch

from .types import PROPERTY_NAMES, PhysicsPropertyTarget, PhysicsTarget

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8

# JSON keys in VLM physical_property → PROPERTY_NAMES index.
_VLM_PROP_KEYS: tuple[str, ...] = (
    "Density (kg/m³)",
    "Young's modulus (MPa)",
    "Poisson's ratio",
)
# Which properties use log(mean) in model space (True) vs raw mean (False).
_VLM_LOG_MEAN: tuple[bool, ...] = (True, True, False)


class PhysicsParser(Protocol):
    """External annotation → typed target. ``target_key`` selects the batch field."""

    target_key: str

    def parse(self, scene: dict, root: Path) -> PhysicsTarget | PhysicsPropertyTarget | None:
        """Parse scene-level physics labels. Return None if absent."""
        ...


# 1-indexed class ids (0 reserved for ignore). Order defines CE class indices.
_3DOVS_CLASS_NAMES: tuple[str, ...] = ("static", "rigid", "soft", "unknown")
_3DOVS_LABEL_TO_ID: dict[str, int] = {
    name: i + 1 for i, name in enumerate(_3DOVS_CLASS_NAMES)
}


def _resolve_labels_path(scene: dict, root: Path) -> Path | None:
    rel = scene.get("physics_labels_path")
    if rel is None:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    return path


class ThreeDOVSJsonParser:
    """Parse ``physics_labels.json``: ``{instance_id_str: class_name, ...}``.

    Manifest field: ``physics_labels_path`` (relative to dataset root or absolute).
    """

    target_key: str = "physics_target"

    def parse(self, scene: dict, root: Path) -> PhysicsTarget | None:
        path = _resolve_labels_path(scene, root)
        if path is None:
            return None
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


def _encode_mean_var(mean: float, variance: float, *, log_mean: bool) -> tuple[float, float]:
    if mean <= 0 and log_mean:
        raise ValueError(f"log-mean property requires mean > 0, got {mean}")
    if variance < 0:
        raise ValueError(f"variance must be >= 0, got {variance}")
    mean_t = math.log(mean) if log_mean else float(mean)
    log_var = math.log(variance + _LOG_EPS)
    return mean_t, log_var


class InstasceneVlmParser:
    """Parse VLM scene JSON list (e.g. Qwen3.6-27B.json) → PhysicsPropertyTarget.

    Each entry: ``{id, response: {physical_property: {...}}, ...}``.
    Instance ``id <= 0`` is ignored (ScanNet id=0 is not a real object).
    """

    target_key: str = "physics_property_target"

    def parse(self, scene: dict, root: Path) -> PhysicsPropertyTarget | None:
        path = _resolve_labels_path(scene, root)
        if path is None:
            return None
        if not path.exists():
            logger.warning("[InstasceneVlmParser] missing %s, skipping", path)
            return None
        with path.open("r") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            return None

        parsed: dict[int, tuple[list[float], list[float]]] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                inst_id = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if inst_id <= 0:
                continue
            response = entry.get("response") or {}
            props = response.get("physical_property")
            if not isinstance(props, dict):
                continue
            means: list[float] = []
            log_vars: list[float] = []
            try:
                for key, use_log in zip(_VLM_PROP_KEYS, _VLM_LOG_MEAN, strict=True):
                    cell = props[key]
                    m, lv = _encode_mean_var(
                        float(cell["mean"]),
                        float(cell["variance"]),
                        log_mean=use_log,
                    )
                    means.append(m)
                    log_vars.append(lv)
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(
                    "[InstasceneVlmParser] skip id=%s incomplete props: %s",
                    inst_id,
                    e,
                )
                continue
            parsed[inst_id] = (means, log_vars)

        if not parsed:
            return None

        max_id = max(parsed.keys())
        p = len(PROPERTY_NAMES)
        mean_lut = torch.zeros(max_id + 1, p, dtype=torch.float32)
        log_var_lut = torch.zeros(max_id + 1, p, dtype=torch.float32)
        valid = torch.zeros(max_id + 1, dtype=torch.bool)
        for inst_id, (means, log_vars) in parsed.items():
            mean_lut[inst_id] = torch.tensor(means, dtype=torch.float32)
            log_var_lut[inst_id] = torch.tensor(log_vars, dtype=torch.float32)
            valid[inst_id] = True
        # id=0 stays False by construction (never written).
        return PhysicsPropertyTarget(
            mean_lut=mean_lut,
            log_var_lut=log_var_lut,
            valid=valid,
            property_names=PROPERTY_NAMES,
        )


PHYSICS_PARSERS: dict[str, PhysicsParser] = {
    "3dovs_json": ThreeDOVSJsonParser(),
    "instascene_vlm": InstasceneVlmParser(),
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

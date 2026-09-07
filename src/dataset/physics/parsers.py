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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import torch

from .types import PROPERTY_NAMES, PhysGMTarget, PhysicsPropertyTarget, PhysicsTarget

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


# --- PhysGM-style normalization (scheme "physgm_copy") ----------------------
# youngs_modulus / poisson_ratio stats are copied verbatim from the PhysGM repo
# (utils.py: E_MEAN/E_STD over log10(Pa), NU_MEAN/NU_STD). Density has no PhysGM
# counterpart; its log10(kg/m³) prior spans foam (~2e1) to metal (~8e3).


@dataclass(frozen=True)
class PhysGMNorm:
    """Raw VLM value → z-scored model space: z = (log10(v·si_scale) - mean) / std."""

    si_scale: float  # VLM unit → SI multiplier
    log10: bool  # apply log10 before z-scoring
    mean: float
    std: float


# PROPERTY_NAMES order: density, youngs_modulus, poisson_ratio.
#
# Fitted 2026-09-07 on the **training split only** (1387 scenes / 49,214
# instances) of the Infinigen VLM pseudo-labels, with the exact transform
# applied below:  scripts/fit_physgm_norm.py --scene_ids
# config/experiment/splits/infinigen_phys_train_ids.txt.  Receipt (n, clamp
# counts, the split it came from):
# config/experiment/splits/physgm_norm_infinigen_train.json.
#
# These replace the constants copied verbatim from the PhysGM repo
# (density 3.0/0.5, E 7.387210/2.456477, nu 0.398/0.111), which were fitted on
# a different corpus and left this one's targets at z ~ (+0.86, 0.54) for E,
# (-0.27, 0.80) for density, (-0.56, 0.60) for nu.  That is not a harmless
# reparameterisation: the head emits (mu, log var) under a Gaussian NLL, so a
# per-property offset and scale error misplaces mu's initial bias, mis-scales
# sigma, and silently reweights the three properties against each other.
# Changing them is only valid because this corpus is the only one in play --
# a different dataset needs its own fit, not these numbers.
#
# ``physgm_denormalize`` below reads the same tuple, so inference-time
# de-normalisation follows automatically; keep it that way (one source, not two).
PHYSGM_NORMALIZATION: tuple[PhysGMNorm, ...] = (
    PhysGMNorm(si_scale=1.0, log10=True, mean=2.863740, std=0.399147),  # density kg/m³
    PhysGMNorm(si_scale=1e6, log10=True, mean=9.495947, std=1.317972),  # E: MPa→Pa
    PhysGMNorm(si_scale=1.0, log10=False, mean=0.336525, std=0.066235),  # nu
)


def physgm_denormalize(values: torch.Tensor) -> torch.Tensor:
    """Model space ``[..., P]`` → SI units (density kg/m³, E Pa, nu). For logging."""
    cols = []
    for i, spec in enumerate(PHYSGM_NORMALIZATION):
        x = values[..., i].float() * spec.std + spec.mean
        cols.append(torch.pow(10.0, x) if spec.log10 else x)
    return torch.stack(cols, dim=-1)


class PhysicsParser(Protocol):
    """External annotation → typed target. ``target_key`` selects the batch field."""

    target_key: str

    def parse(
        self, scene: dict, root: Path
    ) -> PhysicsTarget | PhysicsPropertyTarget | PhysGMTarget | None:
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


def _load_vlm_scene(scene: dict, root: Path, parser_name: str) -> list | None:
    """Read the VLM scene JSON list, or None if missing/empty."""
    path = _resolve_labels_path(scene, root)
    if path is None:
        return None
    if not path.exists():
        logger.warning("[%s] missing %s, skipping", parser_name, path)
        return None
    with path.open("r") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        return None
    return raw


def _iter_vlm_instances(raw: list):
    """Yield ``(inst_id, physical_property dict)`` for well-formed entries (id > 0)."""
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
        yield inst_id, props


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
        raw = _load_vlm_scene(scene, root, type(self).__name__)
        if raw is None:
            return None

        parsed: dict[int, tuple[list[float], list[float]]] = {}
        for inst_id, props in _iter_vlm_instances(raw):
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


# Evaluation-only side table: {scene_id: {inst_id: [z, ...]}} holding the
# training-split class mean for every labelled instance.  Built by
# scripts/build_class_mean_lut.py and discovered by convention next to the
# dataset root, so no config plumbing and no effect when the file is absent.
CLASS_MEAN_LUT_FILENAME = "physgm_class_mean_z.json"


@lru_cache(maxsize=4)
def _load_class_mean_table(root: Path) -> dict[str, dict[str, list[float]]]:
    """Load (once per worker) the class-lookup baseline table under ``root``."""
    path = root / CLASS_MEAN_LUT_FILENAME
    if not path.exists():
        logger.info(
            "[InstasceneVlmPhysGMParser] no %s under %s -- class-lookup baseline "
            "will be unavailable (val/phys_*_clut)",
            CLASS_MEAN_LUT_FILENAME,
            root,
        )
        return {}
    obj = json.loads(path.read_text())
    table = obj.get("by_scene", obj)
    logger.info(
        "[InstasceneVlmPhysGMParser] class-lookup baseline: %d scenes from %s",
        len(table),
        path,
    )
    return table


class InstasceneVlmPhysGMParser:
    """Same VLM scene JSON as InstasceneVlmParser → PhysGMTarget.

    PhysGM-style supervision: z-scored scalars (PHYSGM_NORMALIZATION), no GT
    variance — predictive variance is learned by the Gaussian NLL loss instead.
    """

    target_key: str = "physgm_target"

    def parse(self, scene: dict, root: Path) -> PhysGMTarget | None:
        raw = _load_vlm_scene(scene, root, type(self).__name__)
        if raw is None:
            return None

        parsed: dict[int, list[float]] = {}
        for inst_id, props in _iter_vlm_instances(raw):
            values: list[float] = []
            try:
                for key, spec in zip(_VLM_PROP_KEYS, PHYSGM_NORMALIZATION, strict=True):
                    v = float(props[key]["mean"]) * spec.si_scale
                    if spec.log10:
                        # max(·, 1.0) mirrors PhysGM's log10(max(E, 1.0)).
                        v = math.log10(max(v, 1.0))
                    values.append((v - spec.mean) / spec.std)
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(
                    "[InstasceneVlmPhysGMParser] skip id=%s incomplete props: %s",
                    inst_id,
                    e,
                )
                continue
            parsed[inst_id] = values

        if not parsed:
            return None

        max_id = max(parsed.keys())
        p = len(PROPERTY_NAMES)
        value_lut = torch.zeros(max_id + 1, p, dtype=torch.float32)
        valid = torch.zeros(max_id + 1, dtype=torch.bool)
        for inst_id, values in parsed.items():
            value_lut[inst_id] = torch.tensor(values, dtype=torch.float32)
            valid[inst_id] = True
        # id=0 stays False by construction (never written).
        class_mean_lut = None
        rows = _load_class_mean_table(root).get(str(scene.get("scene_id", "")))
        if rows:
            class_mean_lut = torch.zeros(max_id + 1, p, dtype=torch.float32)
            for inst_id in parsed:
                row = rows.get(str(inst_id))
                if row is not None:
                    class_mean_lut[inst_id] = torch.tensor(row, dtype=torch.float32)

        return PhysGMTarget(
            value_lut=value_lut,
            valid=valid,
            property_names=PROPERTY_NAMES,
            class_mean_lut=class_mean_lut,
        )


PHYSICS_PARSERS: dict[str, PhysicsParser] = {
    "3dovs_json": ThreeDOVSJsonParser(),
    "instascene_vlm": InstasceneVlmParser(),
    "instascene_vlm_physgm": InstasceneVlmPhysGMParser(),
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

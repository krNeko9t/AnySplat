"""Physics supervision types produced by dataset parsers.

Layer contract (dataset → model/loss):
  Parser reads an external annotation format and returns a ``*Target``.
  Downstream code must not re-parse raw JSON / invent class id mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Bool, Float32, Int64
from torch import Tensor

# Canonical property order for the property-regression scheme.
# JSON key → index mapping lives only in parsers.py.
PROPERTY_NAMES: tuple[str, ...] = ("density", "youngs_modulus", "poisson_ratio")


@dataclass
class PhysicsTarget:
    """Scene-level physics **class** supervision (scheme ``class``).

    Conventions (see docs/conventions.md):
      - ``label_lut[instance_id]`` is 1-indexed class id; ``0`` = ignore / unlabeled.
      - ``class_names[i]`` is the human name for CE class ``i`` (0-indexed).
    """

    label_lut: Int64[Tensor, " max_id_plus_1"]
    class_names: tuple[str, ...]


@dataclass
class PhysGMTarget:
    """Scene-level physics **property** supervision, PhysGM-style (scheme ``physgm_copy``).

    Conventions (see docs/conventions.md):
      - ``valid[instance_id]`` False ⇒ ignore (includes id=0 and missing labels).
      - ``value_lut`` holds z-scored model-space scalars (parser-transformed):
          density / youngs_modulus: (log10(raw SI) - mean) / std
          poisson_ratio:            (raw - mean) / std
        Normalization stats live in parsers.py (PHYSGM_NORMALIZATION).
      - Unlike PhysicsPropertyTarget there is no GT variance: the PhysGM loss
        learns predictive variance via Gaussian NLL instead of regressing it.
      - Column order matches ``property_names`` / ``PROPERTY_NAMES``.
    """

    value_lut: Float32[Tensor, "max_id_plus_1 n_prop"]
    valid: Bool[Tensor, " max_id_plus_1"]
    property_names: tuple[str, ...]
    # Optional, evaluation-only: the same rows filled with the **training-split
    # mean of that instance's class** (also z-scored).  This is the trivial
    # class-lookup baseline ticket 03 makes the student's reference frame, and
    # it is carried on the target so the baseline is scored on exactly the
    # instances the student was scored on.  None ⇒ baseline unavailable.
    # Never read by any loss.
    class_mean_lut: Float32[Tensor, "max_id_plus_1 n_prop"] | None = None


@dataclass
class PhysicsPropertyTarget:
    """Scene-level physics **property** supervision (scheme ``property``).

    Conventions (see docs/conventions.md):
      - ``valid[instance_id]`` False ⇒ ignore (includes id=0 and missing labels).
      - ``mean_lut`` / ``log_var_lut`` are in model space (parser-transformed):
          density / youngs_modulus: mean = log(raw_mean), log_var = log(raw_var + eps)
          poisson_ratio: mean = raw_mean, log_var = log(raw_var + eps)
      - Column order matches ``property_names`` / ``PROPERTY_NAMES``.
    """

    mean_lut: Float32[Tensor, "max_id_plus_1 n_prop"]
    log_var_lut: Float32[Tensor, "max_id_plus_1 n_prop"]
    valid: Bool[Tensor, " max_id_plus_1"]
    property_names: tuple[str, ...]

"""Physics supervision types produced by dataset parsers.

Layer contract (dataset → model/loss):
  Parser reads an external annotation format and returns ``PhysicsTarget``.
  Downstream code must not re-parse raw JSON / invent class id mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Int64
from torch import Tensor


@dataclass
class PhysicsTarget:
    """Scene-level physics supervision.

    Conventions (see docs/conventions.md):
      - ``label_lut[instance_id]`` is 1-indexed class id; ``0`` = ignore / unlabeled.
      - ``class_names[i]`` is the human name for CE class ``i`` (0-indexed).
    """

    label_lut: Int64[Tensor, " max_id_plus_1"]
    class_names: tuple[str, ...]

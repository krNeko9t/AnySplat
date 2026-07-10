"""Physics dataset layer: parsers → PhysicsTarget.

Where to edit when requirements change:
  - New annotation format  → ``parsers.py`` (+ register in PHYSICS_PARSERS)
  - Target fields / semantics → ``types.py`` (+ docs/conventions.md)
"""

from .parsers import PHYSICS_PARSERS, get_physics_parser
from .types import PhysicsTarget

__all__ = [
    "PHYSICS_PARSERS",
    "PhysicsTarget",
    "get_physics_parser",
]

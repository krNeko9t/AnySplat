"""Physics dataset layer: parsers → *Target.

Where to edit when requirements change:
  - New annotation format  → ``parsers.py`` (+ register in PHYSICS_PARSERS)
  - Target fields / semantics → ``types.py`` (+ docs/conventions.md)
"""

from .parsers import PHYSICS_PARSERS, get_physics_parser
from .types import PROPERTY_NAMES, PhysicsPropertyTarget, PhysicsTarget

__all__ = [
    "PHYSICS_PARSERS",
    "PROPERTY_NAMES",
    "PhysicsPropertyTarget",
    "PhysicsTarget",
    "get_physics_parser",
]

"""IGGT-style instance / physics heads.

This package provides PartHead, PhysicsHead, PhysicsClassifier,
PhysicsPropertyReadout, and SamProjector.

Where to edit physics:
  - dense features → physics_head.py
  - class logits → physics_classifier.py
  - property readout → physics_property_readout.py
  - pooling → physics_pool.py
"""

from .part_head import PartHead  # noqa: F401
from .physics_classifier import PhysicsClassifier  # noqa: F401
from .physics_head import PhysicsHead  # noqa: F401
from .physics_property_readout import PhysicsPropertyReadout  # noqa: F401
from .sam_projector import SamProjector  # noqa: F401

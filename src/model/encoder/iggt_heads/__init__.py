"""IGGT-style instance / physics heads.

This package provides PartHead, PhysicsHead, PhysicsClassifier, and
SamProjector.  Heavy deps (detectron2, basicsr, sam2) are removed.

Where to edit physics:
  - dense features → physics_head.py
  - instance logits → physics_classifier.py
"""

from .part_head import PartHead  # noqa: F401
from .physics_classifier import PhysicsClassifier  # noqa: F401
from .physics_head import PhysicsHead  # noqa: F401
from .sam_projector import SamProjector  # noqa: F401

"""IGGT-style instance segmentation heads.

This package provides the PartHead (instance embedding head), PhysicsHead
(dense physics feature head), and SamProjector (multi-scale feature adaptor)
ported from the IGGT codebase.  All heavy dependencies (detectron2, basicsr,
sam2) are removed -- the modules are self-contained and only require
PyTorch + xformers.
"""

from .part_head import PartHead  # noqa: F401
from .physics_head import PhysicsHead  # noqa: F401
from .sam_projector import SamProjector  # noqa: F401

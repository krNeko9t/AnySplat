"""PhysicsPropertyPrediction — encoder-side property-regression output slot.

Layer contract (encoder → loss / inference):
  Encoder fills this; loss only reads it. Learnable readout lives in the
  encoder (PhysicsPropertyReadout), not in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import Tensor


@dataclass
class PhysicsPropertyPrediction:
    """Physics property head outputs for one forward pass."""

    # Dense per-pixel features from PhysicsHead: [B, V, C, H, W]
    feat_map: Float[Tensor, "batch view c height width"] | None = None
    # Per-instance (mean, log_var): list length B, each [K_b, P, 2]
    instance_values: list[Float[Tensor, "n_inst n_prop two"]] | None = None
    # Instance ids aligned with instance_values rows: list length B, each [K_b]
    instance_ids: list[Int64[Tensor, " n_inst"]] | None = None
    # Property names matching the P axis (0-indexed columns)
    property_names: tuple[str, ...] | None = None

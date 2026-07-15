"""PhysGMPrediction — encoder-side PhysGM-style property output slot.

Layer contract (encoder → loss / inference):
  Encoder fills this; loss only reads it. Learnable readout lives in the
  encoder (PhysGMReadout), not in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import Tensor


@dataclass
class PhysGMPrediction:
    """PhysGM readout outputs for one forward pass.

    ``mu`` / ``var`` are in normalized model space (see PHYSGM_NORMALIZATION in
    src/dataset/physics/parsers.py). ``var`` is the learned predictive variance
    (softplus-activated, > 0), not a regression of GT variance.
    """

    # Per-instance property means: list length B, each [K_b, P]
    instance_mu: list[Float[Tensor, "n_inst n_prop"]] | None = None
    # Per-instance predictive variances: list length B, each [K_b, P]
    instance_var: list[Float[Tensor, "n_inst n_prop"]] | None = None
    # Instance ids aligned with the rows above: list length B, each [K_b]
    instance_ids: list[Int64[Tensor, " n_inst"]] | None = None
    # Property names matching the P axis (0-indexed columns)
    property_names: tuple[str, ...] | None = None

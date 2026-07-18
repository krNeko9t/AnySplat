"""Encoder-side ``*Prediction`` output slots(EncoderOutput 的可选槽位).

Layer contract (encoder → loss / inference):
  Encoder fills these; loss only reads them. Learnable classifier / readout
  lives in the encoder (PhysicsClassifier / PhysicsPropertyReadout /
  PhysGMReadout), not in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import Tensor


@dataclass
class PhysicsPrediction:
    """Physics head outputs for one forward pass.

    Fields may be None when the corresponding submodule is disabled or when
    instance masks are unavailable (inference without GT / predicted masks).
    """

    # Dense per-pixel features from PhysicsHead: [B, V, C, H, W]
    feat_map: Float[Tensor, "batch view c height width"] | None = None
    # Per-instance class logits from PhysicsClassifier: list length B, each [K_b, num_classes]
    instance_logits: list[Float[Tensor, "n_inst num_classes"]] | None = None
    # Instance ids aligned with instance_logits rows: list length B, each [K_b]
    instance_ids: list[Int64[Tensor, " n_inst"]] | None = None
    # Optional dense per-pixel logits: [B, V, num_classes, H, W]
    dense_logits: Float[Tensor, "batch view num_classes height width"] | None = None
    # CE class names (0-indexed), for logging / export
    class_names: tuple[str, ...] | None = None


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

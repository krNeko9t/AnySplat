"""PhysicsPrediction — encoder-side physics output slot.

Layer contract (encoder → loss / inference):
  Encoder fills this; loss only reads it. Learnable classifier lives in the
  encoder (PhysicsClassifier), not in the loss.
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

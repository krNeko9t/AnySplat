"""层间数据契约:encoder 产出、wrapper/loss 只读。

``EncoderOutput`` 是所有 encoder forward 的统一返回;``*Prediction`` 是其
可选槽位(新增能力 = 新增默认 None 的可选字段,禁止改已有字段语义)。
Learnable classifier / readout 挂在 encoder 侧,不进 loss。
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import Tensor

from src.model.types import Gaussians


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


@dataclass
class SegVGGTPrediction:
    """SegVGGT instance-segmentation branch outputs for one forward pass.

    End-to-end instance reasoning via object queries (no clustering post-process,
    no GT mask pooling): each of the ``Q`` learnable queries yields a per-view mask
    and a class distribution whose last channel is the *no-object* logit, so
    background / empty queries are dropped by argmax rather than a heuristic.

    ``query_embed`` (the projected object-query vectors) is the natural per-object
    instance embedding to hang downstream per-object physics readouts on; it is
    left ``None`` in the inference-only path and populated once needed.
    """

    # Per-query per-view mask logits (pre-sigmoid): [B, Q, V, h, w]
    query_masks: Float[Tensor, "batch query view h w"] | None = None
    # Per-query class logits incl. trailing no-object channel: [B, Q, C_plus_1]
    query_class_logits: Float[Tensor, "batch query classes"] | None = None
    # Projected object-query embeddings (mask / physics space): [B, Q, D]
    query_embed: Float[Tensor, "batch query dim"] | None = None
    # Dense instance feature maps (mask source): [B, V, h, w, D]
    feature_map: Float[Tensor, "batch view h w dim"] | None = None
    # FADA frame-level cross-attention weights (training only): [L, B, Q, V]
    attn_frame_mean: Float[Tensor, "layers batch query view"] | None = None


@dataclass
class EncoderOutput:
    gaussians: Gaussians | None
    pred_pose_enc_list: list[Float[Tensor, "batch view 6"]] | None
    pred_context_pose: dict | None
    depth_dict: dict | None
    infos: dict | None
    distill_infos: dict | None
    # Optional instance segmentation embeddings.
    # instance_feat_map: [B, V, N, H, W] (same spatial resolution as decoder output)
    instance_feat_map: Float[Tensor, "batch view n height width"] | None = None
    # gaussian_instance_feat: [B, G, N] (aligned with gaussians order)
    gaussian_instance_feat: Float[Tensor, "batch gaussian n"] | None = None
    # Physics scheme "class" slot. See PhysicsPrediction.
    physics_prediction: PhysicsPrediction | None = None
    # Physics scheme "property" slot. See PhysicsPropertyPrediction.
    physics_property_prediction: PhysicsPropertyPrediction | None = None
    # Physics scheme "physgm_copy" slot. See PhysGMPrediction.
    physgm_prediction: PhysGMPrediction | None = None
    # SegVGGT end-to-end instance branch. See SegVGGTPrediction.
    segvggt_prediction: SegVGGTPrediction | None = None

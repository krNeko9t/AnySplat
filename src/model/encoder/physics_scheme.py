"""Physics scheme bundles: class / property / physgm_copy wiring for EncoderIGGT.

Switching is by ``phys_scheme`` registry key (Hydra), not bool flags in forward.
Each bundle owns its head + readout, consumes what it needs from
``PhysicsSchemeInputs``, and fills the matching EncoderOutput slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from src.model.heads.physics.physgm_dense_readout import PhysGMDenseReadout
from src.model.heads.physics.physgm_readout import PhysGMReadout
from src.model.heads.physics.physics_classifier import PhysicsClassifier
from src.model.heads.physics.physics_head import PhysicsHead
from src.model.heads.physics.physics_property_readout import PhysicsPropertyReadout
from .predictions import (
    PhysGMPrediction,
    PhysicsPrediction,
    PhysicsPropertyPrediction,
)


@dataclass
class PhysicsSchemeInputs:
    """Everything the encoder exposes to physics schemes (named cross-layer type)."""

    # SamProjector multi-scale maps, 4 × [B*S, C_i, H_i, W_i] (class / property).
    adaptor_features: list[Tensor]
    # Raw aggregator tokens per requested layer, each [B, S, N, token_dim] (physgm_copy).
    aggregated_tokens: list[Tensor]
    images: Tensor  # [B, S, 3, H, W]
    patch_start_idx: int
    patch_size: int
    # Point-head intermediate DPT features (class / property cross-attn).
    point_feature: list[Tensor] | None
    instance_mask: Tensor | None  # [B, S, H, W]
    valid_mask: Tensor | None  # [B, S, H, W] (or [..., 1])


@dataclass
class PhysicsSchemeSlots:
    """Optional EncoderOutput fields produced by a physics scheme."""

    physics_prediction: PhysicsPrediction | None = None
    physics_property_prediction: PhysicsPropertyPrediction | None = None
    physgm_prediction: PhysGMPrediction | None = None


class PhysicsSchemeBundle(Protocol):
    def forward(self, inputs: PhysicsSchemeInputs) -> PhysicsSchemeSlots:
        ...


class ClassPhysicsBundle(nn.Module):
    """PhysicsHead + PhysicsClassifier → physics_prediction."""

    def __init__(
        self,
        *,
        phys_feat_dim: int,
        phys_num_classes: int,
        phys_classifier_hidden: int,
        phys_ignore_id: int,
        phys_dense_logits: bool,
        phys_use_point_feat: bool,
        phys_use_window_cross_attn: bool,
        phys_class_names: tuple[str, ...] | None,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.class_names = phys_class_names
        self.physics_head = PhysicsHead(
            in_channels=[256, 256, 256, 256],
            features=256,
            output_dim=phys_feat_dim,
            patch_size=patch_size,
            window_size=8,
            use_point_feat=phys_use_point_feat,
            use_window_cross_attn=phys_use_window_cross_attn,
        )
        self.physics_classifier = PhysicsClassifier(
            feat_dim=phys_feat_dim,
            num_classes=phys_num_classes,
            hidden=phys_classifier_hidden,
            ignore_id=phys_ignore_id,
            dense_logits=phys_dense_logits,
        )

    def forward(self, inputs: PhysicsSchemeInputs) -> PhysicsSchemeSlots:
        feat_map = self.physics_head(
            inputs.adaptor_features,
            images=inputs.images,
            patch_start_idx=inputs.patch_start_idx,
            point_feature=inputs.point_feature,
        )
        inst_logits = None
        inst_ids = None
        dense_logits = None
        if inputs.instance_mask is not None:
            inst_logits, inst_ids, dense_logits = self.physics_classifier(
                feat_map, inputs.instance_mask, valid_mask=inputs.valid_mask
            )
        return PhysicsSchemeSlots(
            physics_prediction=PhysicsPrediction(
                feat_map=feat_map,
                instance_logits=inst_logits,
                instance_ids=inst_ids,
                dense_logits=dense_logits,
                class_names=self.class_names,
            )
        )


class PropertyPhysicsBundle(nn.Module):
    """PhysicsHead + PhysicsPropertyReadout → physics_property_prediction."""

    def __init__(
        self,
        *,
        phys_feat_dim: int,
        phys_property_hidden: int,
        phys_ignore_id: int,
        phys_use_point_feat: bool,
        phys_use_window_cross_attn: bool,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.physics_head = PhysicsHead(
            in_channels=[256, 256, 256, 256],
            features=256,
            output_dim=phys_feat_dim,
            patch_size=patch_size,
            window_size=8,
            use_point_feat=phys_use_point_feat,
            use_window_cross_attn=phys_use_window_cross_attn,
        )
        self.physics_property_readout = PhysicsPropertyReadout(
            feat_dim=phys_feat_dim,
            hidden=phys_property_hidden,
            ignore_id=phys_ignore_id,
            property_names=PROPERTY_NAMES,
        )

    def forward(self, inputs: PhysicsSchemeInputs) -> PhysicsSchemeSlots:
        feat_map = self.physics_head(
            inputs.adaptor_features,
            images=inputs.images,
            patch_start_idx=inputs.patch_start_idx,
            point_feature=inputs.point_feature,
        )
        inst_values = None
        inst_ids = None
        if inputs.instance_mask is not None:
            inst_values, inst_ids = self.physics_property_readout(
                feat_map, inputs.instance_mask, valid_mask=inputs.valid_mask
            )
        return PhysicsSchemeSlots(
            physics_property_prediction=PhysicsPropertyPrediction(
                feat_map=feat_map,
                instance_values=inst_values,
                instance_ids=inst_ids,
                property_names=PROPERTY_NAMES,
            )
        )


class PhysGMCopyBundle(nn.Module):
    """PhysGMReadout on raw backbone tokens → physgm_prediction.

    No dense head: per-instance tokens are pooled straight from the (frozen)
    aggregator patch tokens, then decoded PhysGM-style into (mu, var).
    """

    def __init__(
        self,
        *,
        token_dim: int,
        physgm_hidden: int,
        phys_ignore_id: int,
    ) -> None:
        super().__init__()
        self.physgm_readout = PhysGMReadout(
            token_dim=token_dim,
            hidden=physgm_hidden,
            ignore_id=phys_ignore_id,
            property_names=PROPERTY_NAMES,
        )

    def forward(self, inputs: PhysicsSchemeInputs) -> PhysicsSchemeSlots:
        if inputs.instance_mask is None:
            return PhysicsSchemeSlots(
                physgm_prediction=PhysGMPrediction(property_names=PROPERTY_NAMES)
            )
        mu, var, ids = self.physgm_readout(
            inputs.aggregated_tokens[-1],
            patch_start_idx=inputs.patch_start_idx,
            images=inputs.images,
            patch_size=inputs.patch_size,
            instance_mask=inputs.instance_mask,
            valid_mask=inputs.valid_mask,
        )
        return PhysicsSchemeSlots(
            physgm_prediction=PhysGMPrediction(
                instance_mu=mu,
                instance_var=var,
                instance_ids=ids,
                property_names=PROPERTY_NAMES,
            )
        )


class PhysGMDPTBundle(nn.Module):
    """PhysicsHead (DPT) + PhysGMDenseReadout → physgm_prediction.

    Dense DPT features at image resolution give the physics path real capacity;
    instances are masked average pools of that map, decoded PhysGM-style into
    (mu, var). Same PhysGMTarget / LossPhysGM supervision as physgm_copy.
    """

    def __init__(
        self,
        *,
        phys_feat_dim: int,
        physgm_hidden: int,
        phys_ignore_id: int,
        phys_use_point_feat: bool,
        phys_use_window_cross_attn: bool,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.physics_head = PhysicsHead(
            in_channels=[256, 256, 256, 256],
            features=256,
            output_dim=phys_feat_dim,
            patch_size=patch_size,
            window_size=8,
            use_point_feat=phys_use_point_feat,
            use_window_cross_attn=phys_use_window_cross_attn,
        )
        self.physgm_dense_readout = PhysGMDenseReadout(
            feat_dim=phys_feat_dim,
            hidden=physgm_hidden,
            ignore_id=phys_ignore_id,
            property_names=PROPERTY_NAMES,
        )

    def forward(self, inputs: PhysicsSchemeInputs) -> PhysicsSchemeSlots:
        if inputs.instance_mask is None:
            return PhysicsSchemeSlots(
                physgm_prediction=PhysGMPrediction(property_names=PROPERTY_NAMES)
            )
        feat_map = self.physics_head(
            inputs.adaptor_features,
            images=inputs.images,
            patch_start_idx=inputs.patch_start_idx,
            point_feature=inputs.point_feature,
        )
        mu, var, ids = self.physgm_dense_readout(
            feat_map, inputs.instance_mask, valid_mask=inputs.valid_mask
        )
        return PhysicsSchemeSlots(
            physgm_prediction=PhysGMPrediction(
                instance_mu=mu,
                instance_var=var,
                instance_ids=ids,
                property_names=PROPERTY_NAMES,
            )
        )


def build_physics_scheme(cfg: Any, *, token_dim: int = 2048) -> nn.Module | None:
    """Factory: ``cfg.phys_scheme`` → bundle module (or None)."""
    scheme = getattr(cfg, "phys_scheme", None)
    if scheme is None:
        return None
    if scheme == "class":
        names = cfg.phys_class_names
        return ClassPhysicsBundle(
            phys_feat_dim=cfg.phys_feat_dim,
            phys_num_classes=cfg.phys_num_classes,
            phys_classifier_hidden=cfg.phys_classifier_hidden,
            phys_ignore_id=cfg.phys_ignore_id,
            phys_dense_logits=cfg.phys_dense_logits,
            phys_use_point_feat=cfg.phys_use_point_feat,
            phys_use_window_cross_attn=cfg.phys_use_window_cross_attn,
            phys_class_names=tuple(names) if names is not None else None,
        )
    if scheme == "property":
        return PropertyPhysicsBundle(
            phys_feat_dim=cfg.phys_feat_dim,
            phys_property_hidden=cfg.phys_property_hidden,
            phys_ignore_id=cfg.phys_ignore_id,
            phys_use_point_feat=cfg.phys_use_point_feat,
            phys_use_window_cross_attn=cfg.phys_use_window_cross_attn,
        )
    if scheme == "physgm_copy":
        return PhysGMCopyBundle(
            token_dim=token_dim,
            physgm_hidden=cfg.physgm_hidden,
            phys_ignore_id=cfg.phys_ignore_id,
        )
    if scheme == "physgm_dpt":
        return PhysGMDPTBundle(
            phys_feat_dim=cfg.phys_feat_dim,
            physgm_hidden=cfg.physgm_hidden,
            phys_ignore_id=cfg.phys_ignore_id,
            phys_use_point_feat=cfg.phys_use_point_feat,
            phys_use_window_cross_attn=cfg.phys_use_window_cross_attn,
        )
    raise ValueError(
        f"Unknown phys_scheme={scheme!r}. "
        "Expected None, 'class', 'property', 'physgm_copy', or 'physgm_dpt'."
    )

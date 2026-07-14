"""Physics scheme bundles: class vs property wiring for EncoderIGGT.

Switching is by ``phys_scheme`` registry key (Hydra), not bool flags in forward.
Each bundle owns PhysicsHead + readout and fills the matching EncoderOutput slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from .iggt_heads.physics_classifier import PhysicsClassifier
from .iggt_heads.physics_head import PhysicsHead
from .iggt_heads.physics_property_readout import PhysicsPropertyReadout
from .physics_prediction import PhysicsPrediction
from .physics_property_prediction import PhysicsPropertyPrediction


@dataclass
class PhysicsSchemeSlots:
    """Optional EncoderOutput fields produced by a physics scheme."""

    physics_prediction: PhysicsPrediction | None = None
    physics_property_prediction: PhysicsPropertyPrediction | None = None


class PhysicsSchemeBundle(Protocol):
    def forward(
        self,
        adaptor_features: list,
        *,
        images: Tensor,
        patch_start_idx: int,
        point_feature: list | None,
        instance_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> PhysicsSchemeSlots:
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

    def forward(
        self,
        adaptor_features: list,
        *,
        images: Tensor,
        patch_start_idx: int,
        point_feature: list | None,
        instance_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> PhysicsSchemeSlots:
        feat_map = self.physics_head(
            adaptor_features,
            images=images,
            patch_start_idx=patch_start_idx,
            point_feature=point_feature,
        )
        inst_logits = None
        inst_ids = None
        dense_logits = None
        if instance_mask is not None:
            inst_logits, inst_ids, dense_logits = self.physics_classifier(
                feat_map, instance_mask, valid_mask=valid_mask
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

    def forward(
        self,
        adaptor_features: list,
        *,
        images: Tensor,
        patch_start_idx: int,
        point_feature: list | None,
        instance_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> PhysicsSchemeSlots:
        feat_map = self.physics_head(
            adaptor_features,
            images=images,
            patch_start_idx=patch_start_idx,
            point_feature=point_feature,
        )
        inst_values = None
        inst_ids = None
        if instance_mask is not None:
            inst_values, inst_ids = self.physics_property_readout(
                feat_map, instance_mask, valid_mask=valid_mask
            )
        return PhysicsSchemeSlots(
            physics_property_prediction=PhysicsPropertyPrediction(
                feat_map=feat_map,
                instance_values=inst_values,
                instance_ids=inst_ids,
                property_names=PROPERTY_NAMES,
            )
        )


def build_physics_scheme(cfg: Any) -> nn.Module | None:
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
    raise ValueError(
        f"Unknown phys_scheme={scheme!r}. Expected None, 'class', or 'property'."
    )

"""IGGT encoder -- VGGT backbone + SamProjector + PartHead (+ optional physics).

Outputs ``EncoderOutput`` with:
  - ``gaussians = None``  (no Gaussian head)
  - ``instance_feat_map``  [B, V, D, H, W]
  - ``physics_prediction`` / ``physics_property_prediction`` / ``physgm_prediction``
    via phys_scheme
  - ``pred_context_pose``  dict with extrinsic / intrinsic
  - ``depth_dict``         dict with depth / world_points / conf

Physics wiring (where to edit):
  - scheme selection → physics_scheme.build_physics_scheme (phys_scheme cfg)
  - dense features  → PhysicsHead
  - class logits    → PhysicsClassifier
  - property values → PhysicsPropertyReadout
  - PhysGM readout  → PhysGMReadout (pools raw aggregator tokens, no dense head)
  - GT labels       → dataset physics parsers (not here)
  - loss formula    → src/loss/loss_phys*.py (not here)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.types import BatchedExample, DataShim
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .encoder import Encoder, EncoderOutput
from .iggt_heads import PartHead, SamProjector
from .physics_scheme import PhysicsSchemeInputs, build_physics_scheme
from .vggt.models.vggt import VGGT

logger = logging.getLogger(__name__)


@dataclass
class EncoderIGGTCfg:
    name: Literal["iggt"]
    instance_feat_dim: int = 8
    pretrained_weights: str = ""
    intermediate_layer_idx: Optional[List[int]] = None
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pred_pose: bool = True
    # Physics scheme: None | "class" | "property" | "physgm_copy" (Hydra switch)
    phys_scheme: Optional[Literal["class", "property", "physgm_copy"]] = None
    phys_feat_dim: int = 32
    phys_ignore_id: int = 0
    phys_use_point_feat: bool = True
    phys_use_window_cross_attn: bool = True
    # scheme == "class"
    phys_num_classes: int = 4
    phys_classifier_hidden: int = 64
    phys_dense_logits: bool = False
    phys_class_names: Optional[List[str]] = None
    # scheme == "property"
    phys_property_hidden: int = 64
    # scheme == "physgm_copy"
    physgm_hidden: int = 64


class EncoderIGGT(Encoder["EncoderIGGTCfg"]):
    """IGGT-style encoder: VGGT backbone + SamProjector + PartHead."""

    def __init__(self, cfg: EncoderIGGTCfg) -> None:
        super().__init__(cfg)

        embed_dim = 1024
        patch_size = 14

        base = VGGT.from_pretrained("facebook/VGGT-1B")
        self.aggregator = base.aggregator.to(torch.bfloat16)
        self.camera_head = base.camera_head
        self.point_head = base.point_head
        self.depth_head = base.depth_head

        self.pred_pose = cfg.pred_pose
        self.instance_feat_dim = cfg.instance_feat_dim
        self._intermediate_layer_idx = cfg.intermediate_layer_idx or [4, 11, 17, 23]

        self.part_adaptor = SamProjector(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            pos_embed=False,
            out_channels=[256, 256, 256, 256],
        )
        self.part_head = PartHead(
            in_channels=[256, 256, 256, 256],
            features=256,
            output_dim=self.instance_feat_dim,
            patch_size=patch_size,
            window_size=8,
        )

        self.physics_scheme = build_physics_scheme(cfg, token_dim=2 * embed_dim)
        self._patch_size = patch_size

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        instance_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> EncoderOutput:
        device = image.device
        b, v, _, h, w = image.shape

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(
                image.to(torch.bfloat16),
                intermediate_layer_idx=self._intermediate_layer_idx,
            )

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )

            pts_all, pts_conf, point_intermediate = self.point_head(
                aggregated_tokens_list,
                images=image,
                patch_start_idx=patch_start_idx,
                return_intermediate=True,
            )

            depth_map, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=image,
                patch_start_idx=patch_start_idx,
            )

        adaptor_out, _pos = self.part_adaptor(
            aggregated_tokens_list,
            images=image,
            patch_start_idx=patch_start_idx,
        )
        point_feat_list = list(point_intermediate) if point_intermediate is not None else None

        instance_feat_map = self.part_head(
            list(adaptor_out.values()),
            images=image,
            patch_start_idx=patch_start_idx,
            point_feature=point_feat_list,
        )
        # hard code normalize for iggt
        instance_feat_map = F.normalize(
            instance_feat_map.float(), p=2, dim=2, eps=1e-8
        ).to(instance_feat_map.dtype)

        physics_prediction = None
        physics_property_prediction = None
        physgm_prediction = None
        if self.physics_scheme is not None:
            slots = self.physics_scheme(
                PhysicsSchemeInputs(
                    adaptor_features=list(adaptor_out.values()),
                    aggregated_tokens=aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                    patch_size=self._patch_size,
                    point_feature=point_feat_list,
                    instance_mask=instance_mask,
                    valid_mask=valid_mask,
                )
            )
            physics_prediction = slots.physics_prediction
            physics_property_prediction = slots.physics_property_prediction
            physgm_prediction = slots.physgm_prediction

        del aggregated_tokens_list, patch_start_idx
        torch.cuda.empty_cache()

        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic_norm = intrinsic.clone()
        intrinsic_norm = torch.stack(
            [intrinsic_norm[:, :, 0] / w, intrinsic_norm[:, :, 1] / h, intrinsic_norm[:, :, 2]],
            dim=2,
        )

        infos = {
            "scene_scale": pts_all.flatten(2, 3).norm(dim=-1).mean().clip(min=1e-8),
            "voxelize_ratio": 1.0,
        }

        return EncoderOutput(
            gaussians=None,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic_norm,
            ),
            depth_dict=dict(
                depth=depth_map,
                depth_conf=depth_conf,
                world_points=pts_all,
                world_points_conf=pts_conf,
            ),
            infos=infos,
            distill_infos=None,
            instance_feat_map=instance_feat_map,
            gaussian_instance_feat=None,
            physics_prediction=physics_prediction,
            physics_property_prediction=physics_property_prediction,
            physgm_prediction=physgm_prediction,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )
            return batch

        return data_shim

"""IGGT encoder -- VGGT backbone + SamProjector + PartHead for instance features.

Mirrors the IGGT architecture from IGGT/iggt/models/vggt.py (the ``IGGT``
class) but uses only local modules so there is no dependency on the IGGT/
subdirectory.

Outputs ``EncoderOutput`` with:
  - ``gaussians = None``  (no Gaussian head)
  - ``instance_feat_map``  [B, V, D, H, W]
  - ``pred_context_pose``  dict with extrinsic / intrinsic
  - ``depth_dict``         dict with depth / world_points / conf
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.types import BatchedExample, DataShim
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .encoder import Encoder, EncoderOutput
from .iggt_heads import PartHead, SamProjector
from .vggt.models.vggt import VGGT

logger = logging.getLogger(__name__)


@dataclass
class EncoderIGGTCfg:
    name: Literal["iggt"]
    instance_feat_dim: int = 8
    freeze_backbone: bool = True
    pretrained_weights: str = ""
    intermediate_layer_idx: Optional[List[int]] = None
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pred_pose: bool = True


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
        self.freeze_backbone = cfg.freeze_backbone
        self.instance_feat_dim = cfg.instance_feat_dim
        self._intermediate_layer_idx = cfg.intermediate_layer_idx or [4, 11, 17, 23]

        if self.freeze_backbone:
            for module in [self.aggregator, self.camera_head, self.point_head, self.depth_head]:
                for param in module.parameters():
                    param.requires_grad = False

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

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
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
        instance_feat_map = self.part_head(
            list(adaptor_out.values()),
            images=image,
            patch_start_idx=patch_start_idx,
            point_feature=list(point_intermediate) if point_intermediate is not None else None,
        )

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

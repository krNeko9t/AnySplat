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
from src.model.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from src.model.outputs import EncoderOutput
from .base import Encoder
from src.model.heads.instance import PartHead, SamProjector
from src.model.heads.physics.scheme import PhysicsSchemeInputs, build_physics_scheme
from src.model.vggt.models.vggt import VGGT

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
    phys_scheme: Optional[Literal["class", "property", "physgm_copy", "physgm_dpt"]] = None
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


def _remap_iggt_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap IGGT checkpoint keys to match the local model structure.

    Handles two transformations:
      1. ``part_head.scratch.X``  ->  ``part_head.X``
      2. Adds ``encoder.`` prefix when keys lack it (raw IGGT ckpt).
    """
    remapped: dict[str, torch.Tensor] = {}
    n_scratch = 0
    n_prefix = 0
    for k, v in state_dict.items():
        new_k = k

        if "part_head.scratch." in new_k:
            new_k = new_k.replace("part_head.scratch.", "part_head.")
            n_scratch += 1

        if not new_k.startswith("encoder."):
            new_k = "encoder." + new_k
            n_prefix += 1

        remapped[new_k] = v

    if n_scratch:
        logger.info("Remapped %d part_head.scratch.* keys", n_scratch)
    if n_prefix:
        logger.info("Added encoder. prefix to %d keys", n_prefix)
    return remapped


def _align_state_dicts(
    model_sd: dict[str, torch.Tensor],
    ckpt_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep only ckpt tensors whose key exists in *model_sd* with matching shape."""
    aligned: dict[str, torch.Tensor] = {}
    matched = mismatched = not_in_ckpt = 0
    for k, v in model_sd.items():
        if k not in ckpt_sd:
            not_in_ckpt += 1
            continue
        ckpt_v = ckpt_sd[k]
        if hasattr(ckpt_v, "shape") and ckpt_v.shape == v.shape:
            aligned[k] = ckpt_v
            matched += 1
        else:
            mismatched += 1
            logger.warning(
                "Shape mismatch: %s  model=%s  ckpt=%s",
                k,
                tuple(v.shape),
                tuple(ckpt_v.shape) if hasattr(ckpt_v, "shape") else "N/A",
            )
    unused = len(ckpt_sd) - matched - mismatched
    logger.info(
        "state_dict aligned: matched=%d, mismatched=%d, not_in_ckpt=%d, unused_in_ckpt=%d",
        matched, mismatched, not_in_ckpt, unused,
    )
    return aligned


class IGGTModel(nn.Module):
    """IGGT model: VGGT backbone + SamProjector + PartHead, no decoder."""

    def __init__(self, encoder_cfg: EncoderIGGTCfg) -> None:
        super().__init__()
        self.encoder = EncoderIGGT(encoder_cfg)

    def forward(
        self,
        context_image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        instance_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[EncoderOutput, None]:
        encoder_output = self.encoder(
            context_image,
            global_step,
            visualization_dump=visualization_dump,
            instance_mask=instance_mask,
            valid_mask=valid_mask,
        )
        return encoder_output, None

    @torch.no_grad()
    def inference(self, context_image: torch.Tensor) -> tuple[EncoderOutput, None]:
        encoder_output = self.encoder(context_image, global_step=0)
        return encoder_output, None

    @classmethod
    def from_checkpoint(
        cls,
        encoder_cfg: EncoderIGGTCfg,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "IGGTModel":
        """Build model and load an IGGT checkpoint with key remapping."""
        model = cls(encoder_cfg)
        raw_sd = torch.load(checkpoint_path, map_location=device)

        if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]
        raw_sd = {k.replace("module.", "", 1): v for k, v in raw_sd.items()}

        remapped = _remap_iggt_checkpoint_keys(raw_sd)
        aligned = _align_state_dicts(model.state_dict(), remapped)

        missing, unexpected = model.load_state_dict(aligned, strict=False)
        logger.info(
            "[IGGTModel.from_checkpoint] loaded %d/%d params (missing=%d, unexpected=%d)",
            len(aligned), len(model.state_dict()), len(missing), len(unexpected),
        )
        return model

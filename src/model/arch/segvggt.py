"""SegVGGT encoder -- end-to-end joint 3D reconstruction + instance segmentation.

Unlike the IGGT route (contrastive instance features + inference-time clustering),
SegVGGT reasons about instances *inside* the geometry-grounded transformer: a set of
learnable object queries cross-attend the image tokens after every global-attention
layer, and each query directly predicts a per-view mask (query .dot. dense feature map)
and a class distribution whose trailing channel is the *no-match* logit (DETR
literature: no-object). Background / empty queries are dropped by argmax -- no
post-processing, no GT-mask pooling.

The whole modified transformer + heads live in the vendored box ``src/model/segvggt/``
(official architecture; class-count arithmetic corrected via
``scannet_instance_taxonomy``). This arch file only:
  - builds the vendored ``SegVGGT`` model from a typed cfg,
  - repackages its prediction dict into the repo-wide ``EncoderOutput`` contract
    (``segvggt_prediction`` slot + geometry slots), and
  - loads the official ``.pt`` checkpoint (key remap, shape-aligned, strict=False).

Training (Hungarian matching, BCE/Dice, FADA JS loss) lives in the loss layer --
``src/loss/loss_segvggt{,_geo}.py`` -- since the official repo ships no training code.

``query_embed`` (the pre-projection object-query vectors) is now exposed: the vendored
forward publishes it under ``instance_queries`` and it is carried through to
``SegVGGTPrediction``. With ``phys_scheme="query_physgm"`` this encoder additionally
runs :class:`QueryPhysGMReadout` on those queries, yielding a per-object (mu, var) for
each physics property. Unlike the IGGT physics path this needs no instance mask, so it
also works at inference on an unseen scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import nn

from src.dataset.physics.types import PROPERTY_NAMES
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.types import BatchedExample, DataShim
from src.model.heads.physics.query_physgm_readout import QueryPhysGMReadout
from src.model.outputs import EncoderOutput, SegVGGTPrediction
from src.model.segvggt.models.segvggt import SegVGGT
from src.model.segvggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.segvggt.utils.scannet_instance_taxonomy import DEFAULT_NON_INSTANCE_CLASSES
from .base import Encoder

logger = logging.getLogger(__name__)


@dataclass
class EncoderSegVGGTCfg:
    name: Literal["segvggt"]
    pretrained_weights: str = ""
    # --- backbone / heads (mirror the official eval config) ---
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    enable_camera: bool = True
    enable_depth: bool = True
    enable_point: bool = False
    enable_track: bool = False
    # Semantic table size (ScanNetv2=20, ScanNet200=200) + which semantic classes
    # are excluded from the instance head.  Head width =
    # (num_semantic_classes - len(non_instance_classes)) + 1 no-match.
    # ScanNet default excludes wall/floor (semantic stuff; no instance ids in GT).
    num_semantic_classes: int = 20
    non_instance_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_NON_INSTANCE_CLASSES)
    )
    return_feature_maps_down_ratio: int = 2
    # --- instance branch (object queries) ---
    enable_instance_seg: bool = True
    instance_query_num: int = 400
    query_self_attention: bool = True
    query_sa_mlp_ratio: float = 1.0
    supervise_attend_frame: bool = True
    # --- LoRA fine-tuning of frame/global attention ---
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_frame_blocks: bool = True
    lora_target_global_blocks: bool = True
    lora_verbose: bool = False
    # --- physics on queries (None = branch absent, weights unchanged) ---
    # Only one scheme exists here, so there is no build_physics_scheme-style factory:
    # "query_physgm" attaches QueryPhysGMReadout to the object queries.
    phys_scheme: Optional[Literal["query_physgm"]] = None
    physgm_hidden: int = 64
    # --- data shim (dataset [-1,1] -> wrapper (x+1)/2 -> [0,1] model input) ---
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


def _build_segvggt(cfg: EncoderSegVGGTCfg) -> SegVGGT:
    lora_config = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_frame_blocks": cfg.lora_target_frame_blocks,
        "target_global_blocks": cfg.lora_target_global_blocks,
        "verbose": cfg.lora_verbose,
    }
    return SegVGGT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        enable_camera=cfg.enable_camera,
        enable_point=cfg.enable_point,
        enable_depth=cfg.enable_depth,
        enable_track=cfg.enable_track,
        num_semantic_classes=cfg.num_semantic_classes,
        non_instance_classes=cfg.non_instance_classes,
        return_feature_maps_down_ratio=cfg.return_feature_maps_down_ratio,
        enable_instance_seg=cfg.enable_instance_seg,
        instance_query_num=cfg.instance_query_num,
        use_lora=cfg.use_lora,
        lora_config=lora_config,
        supervise_attend_frame=cfg.supervise_attend_frame,
        query_self_attention=cfg.query_self_attention,
        query_sa_params={"mlp_ratio": cfg.query_sa_mlp_ratio},
    )


class EncoderSegVGGT(Encoder["EncoderSegVGGTCfg"]):
    """Wraps the vendored SegVGGT model and adapts it to ``EncoderOutput``."""

    def __init__(self, cfg: EncoderSegVGGTCfg) -> None:
        super().__init__(cfg)
        self.model = _build_segvggt(cfg)
        self.query_physgm = None
        if cfg.phys_scheme == "query_physgm":
            self.query_physgm = QueryPhysGMReadout(
                token_dim=cfg.embed_dim,
                hidden=cfg.physgm_hidden,
                property_names=PROPERTY_NAMES,
            )

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        instance_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """``image``: [B, V, 3, H, W] in [0, 1] (aggregator applies ResNet norm)."""
        b, v, _, h, w = image.shape

        # Aggregator runs bf16 on CUDA; the vendored forward already forces the
        # camera/depth heads to fp32 via an inner autocast(enabled=False) block.
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.bfloat16,
            enabled=image.device.type == "cuda",
        ):
            preds = self.model(image)

        pred_pose_enc_list = preds.get("pose_enc_list")
        pred_context_pose = None
        if "pose_enc" in preds:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                preds["pose_enc"].float(), image.shape[-2:]
            )
            extrinsic_padding = (
                torch.tensor([0, 0, 0, 1], device=image.device, dtype=extrinsic.dtype)
                .view(1, 1, 1, 4)
                .repeat(b, v, 1, 1)
            )
            intrinsic_norm = torch.stack(
                [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]],
                dim=2,
            )
            pred_context_pose = dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic_norm,
            )

        depth_dict = None
        if "depth" in preds:
            depth_dict = dict(depth=preds["depth"], depth_conf=preds.get("depth_conf"))

        query_embed = preds.get("instance_queries")          # [B, Q, embed_dim]
        query_phys_mu = query_phys_var = None
        if self.query_physgm is not None and query_embed is not None:
            query_phys_mu, query_phys_var = self.query_physgm(query_embed)

        segvggt_prediction = SegVGGTPrediction(
            query_masks=preds.get("instance_maps"),
            query_class_logits=preds.get("instance_labels"),
            query_embed=query_embed,
            feature_map=preds.get("semantic_feature_maps"),
            attn_frame_mean=preds.get("attn_frame_mean"),
            query_phys_mu=query_phys_mu,
            query_phys_var=query_phys_var,
            property_names=None if self.query_physgm is None else PROPERTY_NAMES,
        )

        return EncoderOutput(
            gaussians=None,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=pred_context_pose,
            depth_dict=depth_dict,
            infos=dict(voxelize_ratio=1.0),
            distill_infos=None,
            segvggt_prediction=segvggt_prediction,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            return apply_normalize_shim(batch, self.cfg.input_mean, self.cfg.input_std)

        return data_shim


def _strip_lightning_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop the ``model.`` prefix a Lightning checkpoint adds, if present.

    ``SegVGGTWrapper`` holds this model as its ``model`` attribute, so a fine-tuned
    checkpoint keys everything as ``model.encoder.model.aggregator...`` while the
    official ``.pt`` uses bare ``aggregator...``.  Stripping the prefix lets both load
    through the same path.

    The ``model.encoder.`` guard is what makes this safe: official checkpoints have no
    such key, so they are returned untouched.
    """
    if not any(k.startswith("model.encoder.") for k in state_dict):
        return state_dict
    stripped = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    logger.info(
        "Detected a Lightning checkpoint; stripped the 'model.' prefix from %d keys",
        sum(1 for k in state_dict if k.startswith("model.")),
    )
    return stripped


def _remap_segvggt_checkpoint_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prefix official SegVGGT keys (``aggregator.*``) to match ``encoder.model.*``."""
    remapped: dict[str, torch.Tensor] = {}
    n_prefix = 0
    for k, v in state_dict.items():
        new_k = k
        if not new_k.startswith("encoder.model."):
            new_k = "encoder.model." + new_k
            n_prefix += 1
        remapped[new_k] = v
    if n_prefix:
        logger.info("Added encoder.model. prefix to %d keys", n_prefix)
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


class SegVGGTModel(nn.Module):
    """SegVGGT model shell: encoder-only (no decoder), mirrors ``IGGTModel``."""

    def __init__(self, encoder_cfg: EncoderSegVGGTCfg) -> None:
        super().__init__()
        self.encoder = EncoderSegVGGT(encoder_cfg)

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
        return self.encoder(context_image, global_step=0), None

    @classmethod
    def from_checkpoint(
        cls,
        encoder_cfg: EncoderSegVGGTCfg,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "SegVGGTModel":
        """Build model and load an official SegVGGT ``.pt`` with key remapping."""
        model = cls(encoder_cfg)
        raw_sd = torch.load(checkpoint_path, map_location=device)

        if isinstance(raw_sd, dict) and "model" in raw_sd:
            raw_sd = raw_sd["model"]
        elif isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]
        raw_sd = {k.replace("module.", "", 1): v for k, v in raw_sd.items()}
        raw_sd = _strip_lightning_prefix(raw_sd)

        remapped = _remap_segvggt_checkpoint_keys(raw_sd)
        aligned = _align_state_dicts(model.state_dict(), remapped)

        missing, unexpected = model.load_state_dict(aligned, strict=False)
        logger.info(
            "[SegVGGTModel.from_checkpoint] loaded %d/%d params (missing=%d, unexpected=%d)",
            len(aligned), len(model.state_dict()), len(missing), len(unexpected),
        )
        return model

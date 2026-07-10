"""IGGT model -- encoder-only architecture for instance feature extraction.

Wraps ``EncoderIGGT`` without a decoder.  Provides checkpoint loading with
key remapping so pre-trained IGGT checkpoints (from the IGGT/ codebase) can
be loaded directly.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.model.encoder.iggt import EncoderIGGT, EncoderIGGTCfg
from src.model.encoder.encoder import EncoderOutput

logger = logging.getLogger(__name__)


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

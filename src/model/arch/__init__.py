import logging
from typing import Optional, Union

from torch import nn

from src.model.decoder import DecoderCfg

from .anysplat import EncoderAnySplatCfg
from .iggt import EncoderIGGTCfg
from .segvggt import EncoderSegVGGTCfg

from .anysplat import AnySplat
from .iggt import IGGTModel
from .segvggt import SegVGGTModel
from .weight_loading import init_anysplat_from_hf

MODELS = {
    "anysplat": AnySplat,
    "iggt": IGGTModel,
    "segvggt": SegVGGTModel,
}

EncoderCfg = Union[EncoderAnySplatCfg, EncoderIGGTCfg, EncoderSegVGGTCfg]

logger = logging.getLogger(__name__)


def get_model(encoder_cfg: EncoderCfg, decoder_cfg: Optional[DecoderCfg] = None) -> nn.Module:
    """Build the model from config, dispatching on ``encoder_cfg.name``."""
    pretrained = getattr(encoder_cfg, "pretrained_weights", "") or ""

    # --- IGGT ---
    if isinstance(encoder_cfg, EncoderIGGTCfg):
        if pretrained and not pretrained.startswith("hf:"):
            model = IGGTModel.from_checkpoint(encoder_cfg, pretrained)
            logger.info("[get_model] IGGT loaded from checkpoint `%s`", pretrained)
            return model
        model = IGGTModel(encoder_cfg)
        logger.info("[get_model] IGGT built (pretrained_weights=%s)", pretrained or "none")
        return model

    # --- SegVGGT ---
    if isinstance(encoder_cfg, EncoderSegVGGTCfg):
        if pretrained and not pretrained.startswith("hf:"):
            model = SegVGGTModel.from_checkpoint(encoder_cfg, pretrained)
            logger.info("[get_model] SegVGGT loaded from checkpoint `%s`", pretrained)
            return model
        model = SegVGGTModel(encoder_cfg)
        logger.info("[get_model] SegVGGT built (pretrained_weights=%s)", pretrained or "none")
        return model

    # --- AnySplat ---
    if isinstance(encoder_cfg, EncoderAnySplatCfg):
        assert decoder_cfg is not None, "AnySplat requires a decoder config"
        if pretrained:
            return init_anysplat_from_hf(pretrained, encoder_cfg, decoder_cfg)
        return AnySplat(encoder_cfg, decoder_cfg)

    raise ValueError(f"Unknown encoder config type: {type(encoder_cfg)}")

import logging
from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.anysplat import EncoderAnySplat, EncoderAnySplatCfg
from ..encoder.iggt import EncoderIGGT, EncoderIGGTCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .anysplat import AnySplat
from .iggt import IGGTModel

MODELS = {
    "anysplat": AnySplat,
    "iggt": IGGTModel,
}

EncoderCfg = Union[EncoderAnySplatCfg, EncoderIGGTCfg]
DecoderCfg = DecoderSplattingCUDACfg

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

    # --- AnySplat ---
    if isinstance(encoder_cfg, EncoderAnySplatCfg):
        assert decoder_cfg is not None, "AnySplat requires a decoder config"
        if pretrained:
            hf_id = pretrained
            if hf_id.startswith("hf:"):
                hf_id = hf_id[len("hf:"):]

            base = AnySplat.from_pretrained(hf_id)
            model = AnySplat(encoder_cfg, decoder_cfg)
            missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
            allowed_missing_prefixes = (
                "encoder.instance_head.",
                "encoder.instance_head_proj.",
                "encoder.part_adaptor.",
                "encoder.part_head.",
            )
            bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
            logger.info("[get_model] Initialized from HF `%s`", hf_id)
            logger.info(
                "[get_model] missing_keys=%d (allowed=%d, unexpected=%d), unexpected_keys=%d",
                len(missing),
                len(missing) - len(bad_missing),
                len(bad_missing),
                len(unexpected),
            )
            if bad_missing:
                prefixes = {}
                for k in bad_missing:
                    p = k.split(".", 2)[:2]
                    p = ".".join(p) + "."
                    prefixes[p] = prefixes.get(p, 0) + 1
                top = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:15]
                logger.info("[get_model] unexpected missing key prefixes (top):")
                for p, c in top:
                    logger.info("  - %s: %d", p, c)
            return model

        return AnySplat(encoder_cfg, decoder_cfg)

    raise ValueError(f"Unknown encoder config type: {type(encoder_cfg)}")

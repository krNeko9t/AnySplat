from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.anysplat import EncoderAnySplat, EncoderAnySplatCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .anysplat import AnySplat

MODELS = {
    "anysplat": AnySplat,
}

EncoderCfg = Union[EncoderAnySplatCfg]
DecoderCfg = DecoderSplattingCUDACfg


# hard code for now
def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    # Optional initialization from a Hugging Face AnySplat checkpoint.
    # We keep this at model-construction time (instead of Lightning checkpointing)
    # so users can start from `lhjiang/anysplat` while adding extra heads.
    pretrained = getattr(encoder_cfg, "pretrained_weights", "") or ""
    if isinstance(encoder_cfg, EncoderAnySplatCfg) and pretrained:
        hf_id = pretrained
        if hf_id.startswith("hf:"):
            hf_id = hf_id[len("hf:") :]

        base = AnySplat.from_pretrained(hf_id)
        model = MODELS["anysplat"](encoder_cfg, decoder_cfg)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        print(
            f"[get_model] Initialized from HF `{hf_id}`; "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )
        return model

    return MODELS["anysplat"](encoder_cfg, decoder_cfg)

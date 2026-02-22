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
        allowed_missing_prefixes = (
            "encoder.instance_head.",
            "encoder.instance_head_proj.",
            "encoder.part_adaptor.",
            "encoder.part_head.",
        )
        bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        print(f"[get_model] Initialized from HF `{hf_id}`")
        print(
            f"[get_model] missing_keys={len(missing)} "
            f"(allowed={len(missing) - len(bad_missing)}, unexpected={len(bad_missing)}), "
            f"unexpected_keys={len(unexpected)}"
        )
        if bad_missing:
            prefixes = {}
            for k in bad_missing:
                p = k.split(".", 2)[:2]
                p = ".".join(p) + "."
                prefixes[p] = prefixes.get(p, 0) + 1
            top = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:15]
            print("[get_model] unexpected missing key prefixes (top):")
            for p, c in top:
                print(f"  - {p}: {c}")
        return model

    return MODELS["anysplat"](encoder_cfg, decoder_cfg)

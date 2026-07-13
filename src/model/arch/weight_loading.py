"""Shared weight-loading helpers for arch models and inference scripts.

Authority for pretrained / Lightning-ckpt loading lives here (and in
``get_model`` / ``IGGTModel.from_checkpoint``). Scripts must call these
helpers instead of re-implementing HF + strict=False or Lightning peel logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch
from torch import nn

from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from ..encoder.anysplat import EncoderAnySplatCfg
from .anysplat import AnySplat

logger = logging.getLogger(__name__)

# New randomly-initialized heads may be missing when loading HF AnySplat weights.
# Add a prefix here when introducing a new head; do not copy this list into scripts.
ALLOWED_ANYSPLAT_MISSING_PREFIXES: tuple[str, ...] = (
    "encoder.instance_head.",
    "encoder.instance_head_proj.",
    "encoder.part_adaptor.",
    "encoder.part_head.",
)


def load_lightning_state_dict(ckpt_path: Union[str, Path]) -> dict:
    """Peel ``state_dict`` from a Lightning ``.ckpt`` (or return a bare dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def init_anysplat_from_hf(
    hf_id: str,
    encoder_cfg: EncoderAnySplatCfg,
    decoder_cfg: DecoderSplattingCUDACfg,
    *,
    base: AnySplat | None = None,
) -> AnySplat:
    """Build AnySplat with ``encoder_cfg``/``decoder_cfg``, fill weights from HF.

    Pass an already-loaded ``base`` to avoid a second Hub download when the
    caller needed to inspect/override ``base.encoder_cfg`` first.

    Missing keys under ``ALLOWED_ANYSPLAT_MISSING_PREFIXES`` are expected
    (new heads). Other missing keys are logged as unexpected.
    """
    if hf_id.startswith("hf:"):
        hf_id = hf_id[len("hf:") :]

    own_base = base is None
    if own_base:
        base = AnySplat.from_pretrained(hf_id)
    assert base is not None

    model = AnySplat(encoder_cfg, decoder_cfg)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if own_base:
        del base

    bad_missing = [
        k for k in missing if not k.startswith(ALLOWED_ANYSPLAT_MISSING_PREFIXES)
    ]
    # Print as well as log: CLI scripts often have no logging config, so
    # logger.info alone would hide the only strict=False safety signal.
    msg = (
        f"[init_anysplat_from_hf] HF `{hf_id}` "
        f"missing={len(missing)} (allowed={len(missing) - len(bad_missing)}, "
        f"bad={len(bad_missing)}), unexpected={len(unexpected)}"
    )
    logger.info(msg)
    print(msg)
    if bad_missing:
        prefixes: dict[str, int] = {}
        for k in bad_missing:
            p = ".".join(k.split(".", 2)[:2]) + "."
            prefixes[p] = prefixes.get(p, 0) + 1
        top = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:15]
        header = "[init_anysplat_from_hf] unexpected missing key prefixes (top):"
        logger.info(header)
        print(header)
        for p, c in top:
            line = f"  - {p}: {c}"
            logger.info(line)
            print(line)
    return model


def load_model_from_run(
    run_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Load arch model from a training run's Hydra config + Lightning ckpt.

    Returns ``wrapper.model`` (AnySplat or IGGTModel). Callers that only need
    the encoder should take ``.encoder``.
    """
    from omegaconf import OmegaConf

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.arch import get_model
    from src.model.encoder.iggt import EncoderIGGTCfg

    run_dir = Path(run_dir)
    ckpt_path = Path(ckpt_path)
    cfg_dict = OmegaConf.load(str(run_dir / ".hydra" / "config.yaml"))
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    decoder_cfg = getattr(cfg.model, "decoder", None)
    model = get_model(cfg.model.encoder, decoder_cfg)
    step_tracker = StepTracker()

    if isinstance(cfg.model.encoder, EncoderIGGTCfg):
        from src.model.iggt_wrapper import IGGTWrapper

        wrapper = IGGTWrapper(
            cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), step_tracker,
        )
    else:
        from src.model.anysplat_wrapper import AnySplatWrapper

        wrapper = AnySplatWrapper(
            cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), step_tracker,
        )

    state_dict = load_lightning_state_dict(ckpt_path)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    # Print as well as log: CLI scripts often have no logging config, so
    # logger.info alone would hide the only strict=False safety signal.
    msg = (
        f"[load_model_from_run] ckpt={ckpt_path} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    logger.info(msg)
    print(msg)

    wrapper = wrapper.to(device).eval()
    return wrapper.model

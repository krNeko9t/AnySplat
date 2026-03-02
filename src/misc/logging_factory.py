"""Factory for creating the configured Lightning logger (wandb / tensorboard / local)."""

from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from .LocalLogger import LocalLogger
from .tb_logger import TBLogger


def create_logger(cfg: Union[DictConfig, dict], output_dir: Path):
    """Create a Lightning Logger from config.

    Uses cfg.logger if set ("wandb" | "tensorboard" | "local"). If cfg.logger is not
    set, falls back to wandb.mode != "disabled" for backward compatibility.
    """
    # logger_type = OmegaConf.get(cfg, "logger", None)
    logger_type = cfg.get("logger", None)
    if logger_type is None:
        # Backward compatibility: same as previous "if wandb.mode != disabled"
        logger_type = "wandb" if cfg.wandb.get("mode", "disabled") != "disabled" else "local"

    logger_type = str(logger_type).strip().lower()
    exp_name = cfg.wandb.get("name", "exp")

    if logger_type == "wandb":
        from lightning.pytorch.loggers.wandb import WandbLogger

        return WandbLogger(
            project=cfg.wandb.get("project", "anysplat"),
            mode=cfg.wandb.get("mode", "online"),
            name=f"{exp_name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg) if hasattr(cfg, "wandb") else cfg,
        )
    if logger_type == "tensorboard":
        return TBLogger(
            save_dir=output_dir,
            name=exp_name,
            version=None,
        )
    # "local" or any unknown value
    return LocalLogger()

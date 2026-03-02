"""TensorBoard logger adapter with log_image compatible with WandbLogger/LocalLogger usage."""

from typing import Any, Optional

import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only


class TBLogger(TensorBoardLogger):
    """TensorBoardLogger that implements log_image so ModelWrapper can call it unchanged."""

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if step is None:
            return
        for index, image in enumerate(images):
            if isinstance(image, np.ndarray):
                # (H, W, C) -> (C, H, W) for add_image
                if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                    img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                else:
                    img = torch.from_numpy(image).float()
                if img.dim() == 2:
                    img = img.unsqueeze(0)
            elif isinstance(image, torch.Tensor):
                img = image.detach().float()
                if img.dim() == 3 and img.shape[0] not in (1, 3, 4):
                    img = img.permute(2, 0, 1)
                if img.max() > 1.0:
                    img = img / 255.0
                if img.dim() == 2:
                    img = img.unsqueeze(0)
            else:
                continue
            tag = f"{key}/{index}" if len(images) > 1 else key
            self.experiment.add_image(
                tag=tag,
                img_tensor=img.clamp(0.0, 1.0),
                global_step=step,
                dataformats="CHW",
            )

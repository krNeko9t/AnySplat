"""PhysicsClassifier — instance pooling + MLP over PhysicsHead features.

Lives on the encoder (learnable params). Loss must not own this module.

Call sites:
  - training / val with GT masks: EncoderIGGT.forward(..., instance_mask=...)
  - inference with predicted masks: encoder.physics_classifier(feat_map, masks)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .physics_pool import pool_one_sample


class PhysicsClassifier(nn.Module):
    """Masked average-pool dense physics features → per-instance logits."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden: int = 64,
        ignore_id: int = 0,
        dense_logits: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )
        self.dense_proj: nn.Conv2d | None
        if dense_logits:
            self.dense_proj = nn.Conv2d(feat_dim, num_classes, kernel_size=1)
        else:
            self.dense_proj = None

    def forward(
        self,
        feat_map: Tensor,
        instance_mask: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], Tensor | None]:
        """Classify instances present in ``instance_mask``.

        Parameters
        ----------
        feat_map : [B, V, C, H, W]
        instance_mask : [B, V, H, W]
        valid_mask : optional [B, V, H, W] (or [B, V, H, W, 1])

        Returns
        -------
        instance_logits : list length B, each [K_b, num_classes]
        instance_ids : list length B, each [K_b] (raw instance ids, ignore_id excluded)
        dense_logits : [B, V, num_classes, H, W] or None
        """
        B, V, C, H, W = feat_map.shape
        logits_list: list[Tensor] = []
        ids_list: list[Tensor] = []

        for b in range(B):
            feat_b = feat_map[b]
            mask_b = instance_mask[b]
            vm_b = valid_mask[b] if valid_mask is not None else None
            pooled, ids = pool_one_sample(
                feat_b, mask_b, vm_b, ignore_id=self.ignore_id
            )
            if pooled.shape[0] == 0:
                logits_list.append(
                    torch.empty(
                        (0, self.num_classes),
                        device=feat_map.device,
                        dtype=torch.float32,
                    )
                )
                ids_list.append(
                    torch.empty((0,), device=feat_map.device, dtype=torch.long)
                )
            else:
                logits_list.append(self.mlp(pooled.float()))
                ids_list.append(ids)

        dense = None
        if self.dense_proj is not None:
            dense = self.dense_proj(feat_map.reshape(B * V, C, H, W))
            dense = dense.reshape(B, V, self.num_classes, H, W)

        return logits_list, ids_list, dense

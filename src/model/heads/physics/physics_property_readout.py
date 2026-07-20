"""PhysicsPropertyReadout — pool PhysicsHead features → per-instance properties.

Predicts mean and log_variance for each property in PROPERTY_NAMES order.
Lives on the encoder (learnable). Loss must not own this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from .physics_pool import pool_one_sample


class PhysicsPropertyReadout(nn.Module):
    """Masked average-pool → MLP → [K, P, 2] (mean, log_var)."""

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 64,
        ignore_id: int = 0,
        property_names: tuple[str, ...] = PROPERTY_NAMES,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.ignore_id = ignore_id
        self.property_names = property_names
        self.num_properties = len(property_names)
        out_dim = self.num_properties * 2
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        feat_map: Tensor,
        instance_mask: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """

        Returns
        -------
        instance_values : list length B, each [K_b, P, 2]
        instance_ids : list length B, each [K_b]
        """
        B = feat_map.shape[0]
        values_list: list[Tensor] = []
        ids_list: list[Tensor] = []
        p = self.num_properties

        for b in range(B):
            vm_b = valid_mask[b] if valid_mask is not None else None
            pooled, ids = pool_one_sample(
                feat_map[b],
                instance_mask[b],
                vm_b,
                ignore_id=self.ignore_id,
            )
            if pooled.shape[0] == 0:
                values_list.append(
                    torch.empty(
                        (0, p, 2), device=feat_map.device, dtype=torch.float32
                    )
                )
                ids_list.append(
                    torch.empty((0,), device=feat_map.device, dtype=torch.long)
                )
            else:
                # Force fp32 readout (repo convention: heads/loss outside bf16 autocast).
                # .float() alone is not enough under Lightning bf16-mixed — Linear stays
                # autocast-eligible and still runs in bf16, which breaks backward.
                with torch.amp.autocast("cuda", enabled=False):
                    flat = self.mlp(pooled.float())
                values_list.append(flat.view(-1, p, 2))
                ids_list.append(ids)

        return values_list, ids_list

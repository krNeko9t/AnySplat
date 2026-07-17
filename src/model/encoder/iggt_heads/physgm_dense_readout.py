"""PhysGMDenseReadout — pool PhysicsHead dense features → PhysGM (mu, var).

Scheme "physgm_dpt": the DPT PhysicsHead provides capacity and a dense feature
map at image resolution; each instance is a masked average pool of that map,
decoded per property by the PhysGM MLP (LayerNorm → Linear → GELU → Linear(·, 2))
into (mu, softplus var). Supervision is identical to scheme "physgm_copy".

Lives on the encoder (learnable). Loss must not own this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from .physgm_readout import _make_property_decoder
from .physics_pool import pool_one_sample


class PhysGMDenseReadout(nn.Module):
    """Masked average-pool of dense features → per-property (mu, var)."""

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
        self.decoders = nn.ModuleList(
            _make_property_decoder(feat_dim, hidden) for _ in property_names
        )

    def forward(
        self,
        feat_map: Tensor,
        instance_mask: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """
        Parameters
        ----------
        feat_map : [B, V, C, H, W] PhysicsHead output (image resolution).
        instance_mask : [B, V, H, W]; valid_mask optional [B, V, H, W] (or [..., 1]).

        Returns
        -------
        mu_list, var_list : list length B, each [K_b, P]
        ids_list : list length B, each [K_b]
        """
        B = feat_map.shape[0]
        p = len(self.property_names)

        mu_list: list[Tensor] = []
        var_list: list[Tensor] = []
        ids_list: list[Tensor] = []

        for b in range(B):
            vm_b = valid_mask[b] if valid_mask is not None else None
            pooled, ids = pool_one_sample(
                feat_map[b].float(),  # pool in fp32 under bf16-mixed
                instance_mask[b],
                vm_b,
                ignore_id=self.ignore_id,
            )
            # Force fp32 readout (repo convention: heads/loss outside bf16 autocast).
            # Empty pool: still run a dummy decode and slice [:0] so the [0, P]
            # tensors keep a grad_fn (bare torch.empty has none → loss backward crash).
            with torch.amp.autocast("cuda", enabled=False):
                if pooled.shape[0] == 0:
                    pooled = feat_map[b].float().reshape(feat_map.shape[2], -1).mean(-1).unsqueeze(0)
                    outs = [decoder(pooled) for decoder in self.decoders]
                    out = torch.stack(outs, dim=1)  # [1, P, 2]
                    mu_list.append(out[..., 0][:0])
                    var_list.append((F.softplus(out[..., 1]) + 1e-2)[:0])
                    ids_list.append(
                        torch.empty((0,), device=feat_map.device, dtype=torch.long)
                    )
                    continue
                pooled = pooled.float()
                outs = [decoder(pooled) for decoder in self.decoders]  # P × [K, 2]
            out = torch.stack(outs, dim=1)  # [K, P, 2]
            mu_list.append(out[..., 0])
            var_list.append(F.softplus(out[..., 1]) + 1e-2)
            ids_list.append(ids)

        return mu_list, var_list, ids_list

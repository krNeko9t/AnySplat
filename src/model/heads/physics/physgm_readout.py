"""PhysGMReadout — PhysGM-style per-instance property readout.

Faithful adaptation of PhysGM (model/physgm.py) to instance-level prediction:
PhysGM pools the whole scene into learnable global tokens and decodes each
property with a tiny MLP; here the "token" for each instance is a masked
average pool of the frozen backbone patch tokens, decoded by the same MLP
(LayerNorm → Linear → GELU → Linear(·, 2)) into (mu, softplus var).

No dense feature head — the physics path adds no full-resolution activations.
Lives on the encoder (learnable). Loss must not own this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from .physics_pool import pool_one_sample


def _make_property_decoder(token_dim: int, hidden: int) -> nn.Sequential:
    """PhysGM E/nu token decoder: LayerNorm → Linear → GELU → Linear(·, 2)."""
    decoder = nn.Sequential(
        nn.LayerNorm(token_dim, bias=False),
        nn.Linear(token_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, 2, bias=True),
    )
    # PhysGM init: small last layer so mu/var start near (0, softplus(0)).
    nn.init.normal_(decoder[-1].weight, std=0.01)
    nn.init.constant_(decoder[-1].bias, 0.0)
    return decoder


class PhysGMReadout(nn.Module):
    """Masked average-pool of backbone patch tokens → per-property (mu, var)."""

    def __init__(
        self,
        token_dim: int,
        hidden: int = 64,
        ignore_id: int = 0,
        property_names: tuple[str, ...] = PROPERTY_NAMES,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.ignore_id = ignore_id
        self.property_names = property_names
        self.decoders = nn.ModuleList(
            _make_property_decoder(token_dim, hidden) for _ in property_names
        )

    def forward(
        self,
        tokens: Tensor,
        patch_start_idx: int,
        images: Tensor,
        patch_size: int,
        instance_mask: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """
        Parameters
        ----------
        tokens : [B, S, N, C] aggregator tokens (patch tokens from patch_start_idx).
        images : [B, S, 3, H, W] (shape inference only).
        instance_mask : [B, S, H, W]; valid_mask optional [B, S, H, W] (or [..., 1]).

        Returns
        -------
        mu_list, var_list : list length B, each [K_b, P]
        ids_list : list length B, each [K_b]
        """
        B, S, _, H, W = images.shape
        ph, pw = H // patch_size, W // patch_size
        p = len(self.property_names)

        patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, ph*pw, C]
        feat = patch_tokens.reshape(B, S, ph, pw, -1).permute(0, 1, 4, 2, 3)  # [B,S,C,ph,pw]

        inst_small = (
            F.interpolate(instance_mask.float(), size=(ph, pw), mode="nearest")
            .long()
        )  # [B, S, ph, pw]
        valid_small = None
        if valid_mask is not None:
            if valid_mask.ndim == 5 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            valid_small = (
                F.interpolate(valid_mask.float(), size=(ph, pw), mode="nearest") > 0.5
            )

        mu_list: list[Tensor] = []
        var_list: list[Tensor] = []
        ids_list: list[Tensor] = []

        for b in range(B):
            pooled, ids = pool_one_sample(
                feat[b].float(),  # pool in fp32 (backbone tokens are bf16)
                inst_small[b],
                valid_small[b] if valid_small is not None else None,
                ignore_id=self.ignore_id,
            )
            # Force fp32 readout (repo convention: heads/loss outside bf16 autocast).
            # Empty pool: dummy decode + [:0] keeps grad_fn on empty [0, P] outputs.
            with torch.amp.autocast("cuda", enabled=False):
                if pooled.shape[0] == 0:
                    pooled = feat[b].float().reshape(feat.shape[2], -1).mean(-1).unsqueeze(0)
                    outs = [decoder(pooled) for decoder in self.decoders]
                    out = torch.stack(outs, dim=1)  # [1, P, 2]
                    mu_list.append(out[..., 0][:0])
                    var_list.append((F.softplus(out[..., 1]) + 1e-2)[:0])
                    ids_list.append(
                        torch.empty((0,), device=tokens.device, dtype=torch.long)
                    )
                    continue
                pooled = pooled.float()
                outs = [decoder(pooled) for decoder in self.decoders]  # P × [K, 2]
            out = torch.stack(outs, dim=1)  # [K, P, 2]
            mu_list.append(out[..., 0])
            # PhysGM variance activation: softplus + floor.
            var_list.append(F.softplus(out[..., 1]) + 1e-2)
            ids_list.append(ids)

        return mu_list, var_list, ids_list

"""SamProjector – multi-scale feature adaptor for the PartHead.

Adapted from IGGT/iggt/heads/adaptor.py (SamProjector).  Dependencies on
detectron2 (ShapeSpec) and sam2 (PositionEmbeddingSine) are removed; a
lightweight sinusoidal positional encoding is inlined instead.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight positional encoding (replaces sam2.PositionEmbeddingSine)
# ---------------------------------------------------------------------------


class PositionEmbeddingSine2D(nn.Module):
    """2-D sinusoidal positional encoding (similar to SAM2)."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> pos: (B, 2*num_pos_feats, H, W)."""
        _, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)
        y_embed = y_embed / (H + 1e-6) * 2 * math.pi
        x_embed = x_embed / (W + 1e-6) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # H, W, D
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # C, H, W
        return pos.unsqueeze(0).expand(x.shape[0], -1, -1, -1)


# ---------------------------------------------------------------------------
# Residual projection block (same as IGGT's Projects)
# ---------------------------------------------------------------------------


class _ResidualProjection(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        self.output_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x
        x = self.residual_conv(x)
        x = x + residual
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# SamProjector
# ---------------------------------------------------------------------------


class SamProjector(nn.Module):
    """Extract multi-scale features from aggregated VGGT tokens.

    Returns a dict of 4 feature maps at different resolutions, plus
    positional encodings for each.
    """

    intermediate_layer_idx: List[int]

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        pos_embed: bool = False,
        intermediate_layer_idx: List[int] | None = None,
        out_channels: List[int] | None = None,
    ) -> None:
        super().__init__()
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]
        if out_channels is None:
            out_channels = [256, 256, 256, 256]

        self.out_channels = out_channels
        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = patch_size
        self.pos_embed = pos_embed

        self.norm = nn.LayerNorm(dim_in)

        # 1×1 projection from token dim to per-scale channel dim.
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        # Resize layers + residual projection blocks (matches IGGT's SamProjector).
        self.resize_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=2, padding=1),
                _ResidualProjection(out_channels[0], out_channels[0]),
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=2, padding=1),
                _ResidualProjection(out_channels[0], out_channels[0]),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                _ResidualProjection(out_channels[1], out_channels[1]),
            ),
            nn.Sequential(
                nn.Identity(),
                _ResidualProjection(out_channels[2], out_channels[2]),
            ),
            nn.Sequential(
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
                _ResidualProjection(out_channels[3], out_channels[3]),
            ),
        ])

        self.pes = PositionEmbeddingSine2D(num_pos_feats=out_channels[0] // 2)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out: Dict[str, torch.Tensor] = {}
        pos: Dict[str, torch.Tensor] = {}
        keys = ["res1", "res2", "res3", "res4"]

        for dpt_idx, (layer_idx, key) in enumerate(zip(self.intermediate_layer_idx, keys)):
            # AnySplat's aggregator returns only the requested layers (short
            # list) when called with intermediate_layer_idx; IGGT's aggregator
            # always returns all 24 layers.  Detect which case we have.
            if len(aggregated_tokens_list) > max(self.intermediate_layer_idx):
                # Full list – index by layer number (IGGT path).
                x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            else:
                # Filtered list – index by position (AnySplat path).
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = aggregated_tokens_list[list_idx][:, :, patch_start_idx:]
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            x = self.resize_layers[dpt_idx](x)
            out[key] = x
            pos[key] = self.pes(x)

        return out, pos

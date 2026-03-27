"""PhysicsHead -- dense per-pixel physics feature head.

Mirrors the PartHead architecture (DPT-style multi-scale fusion with optional
cross-attention on point-head intermediate features) but outputs a dense
physics feature map instead of instance embeddings.  The per-instance pooling
and classification happen in the loss module so that GT masks are not needed
at the encoder level.

Output shape: [B, S, phys_feat_dim, H, W]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_blocks import MemEffCrossAttention
from .window_attention import SwinCA, SwinSA


# ---------------------------------------------------------------------------
# DPT fusion helpers (shared with PartHead)
# ---------------------------------------------------------------------------


def _custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    n_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if n_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(n_elements // INT_MAX) + 1, dim=0)
        return torch.cat(
            [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks],
            dim=0,
        ).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class _ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True, align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.has_residual = has_residual
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        if has_residual:
            self.resConfUnit1 = _ResidualConvUnit(features)
        self.resConfUnit2 = _ResidualConvUnit(features)

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if size is None:
            modifier = {"scale_factor": 2}
        else:
            modifier = {"size": size}
        output = _custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


# ---------------------------------------------------------------------------
# PhysicsHead
# ---------------------------------------------------------------------------


class PhysicsHead(nn.Module):
    """Dense physics feature head (symmetric to PartHead).

    Parameters
    ----------
    in_channels : list[int]
        Channel dims of the 4 multi-scale feature maps from SamProjector.
    features : int
        Internal feature dimension for DPT fusion.
    output_dim : int
        Embedding dimension of the output per-pixel physics features.
    patch_size : int
        VGGT patch size (used for resolution calculations).
    window_size : int
        Window size for SwinSA / SwinCA.
    use_point_feat : bool
        Whether to use cross-attention with point_intermediate features.
    use_window_cross_attn : bool
        Whether to use SwinCA window cross-attention at one fusion scale.
    """

    def __init__(
        self,
        in_channels: List[int] | None = None,
        features: int = 256,
        output_dim: int = 32,
        patch_size: int = 14,
        window_size: int = 8,
        use_point_feat: bool = True,
        use_window_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [256, 256, 256, 256]
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.use_point_feat = use_point_feat
        self.use_window_cross_attn = use_window_cross_attn

        head_features_1 = features
        head_features_2 = 32

        self.layer1_rn = nn.Conv2d(in_channels[0], features, 3, 1, 1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], features, 3, 1, 1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], features, 3, 1, 1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], features, 3, 1, 1, bias=False)

        self.refinenet1 = _FeatureFusionBlock(features, has_residual=True)
        self.refinenet2 = _FeatureFusionBlock(features, has_residual=True)
        self.refinenet3 = _FeatureFusionBlock(features, has_residual=True)
        self.refinenet4 = _FeatureFusionBlock(features, has_residual=False)

        self.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        conv2_in_channels = head_features_1 // 2
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        if use_point_feat:
            self.cross_attention_1 = MemEffCrossAttention(
                dim=head_features_1, num_heads=8, qkv_bias=True,
            )
            self.cross_attention_2 = MemEffCrossAttention(
                dim=head_features_1, num_heads=8, qkv_bias=True,
            )

        self.window_self_atten = SwinSA(
            img_size=512,
            out_chans=conv2_in_channels,
            embed_dim=conv2_in_channels,
            num_heads=4,
            window_size=window_size,
        )

        if use_point_feat and use_window_cross_attn:
            self.window_cross_attention = SwinCA(
                img_size=128,
                out_chans=head_features_1,
                embed_dim=head_features_1,
                num_heads=4,
                window_size=window_size,
            )

    # ------------------------------------------------------------------

    def _scratch_forward(
        self,
        features: List[torch.Tensor],
        point_feat: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        out = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn

        if self.use_point_feat and point_feat is not None:
            out4 = out.flatten(2).permute(0, 2, 1)
            pf4 = point_feat[2].flatten(2).permute(0, 2, 1)
            out4 = self.cross_attention_2(out4, pf4, pf4)
            out4 = out4.permute(0, 2, 1).view_as(out)
        else:
            out4 = out

        out = self.refinenet3(out4, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn

        if self.use_point_feat and point_feat is not None:
            out3 = out.flatten(2).permute(0, 2, 1)
            pf3 = point_feat[1].flatten(2).permute(0, 2, 1)
            out3 = self.cross_attention_1(out3, pf3, pf3)
            out3 = out3.permute(0, 2, 1).view_as(out)

        out = self.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn

        if self.use_point_feat and self.use_window_cross_attn and point_feat is not None:
            out2 = out.permute(0, 2, 3, 1)
            pf2 = point_feat[0].permute(0, 2, 3, 1)
            out2 = self.window_cross_attention(out2, pf2, pf2)
            out2 = out2.permute(0, 3, 1, 2)
        else:
            out2 = out

        out = self.refinenet1(out2, layer_1_rn)
        del layer_1_rn

        out = self.output_conv1(out)
        return out

    # ------------------------------------------------------------------

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        point_feature: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        multi_scale_features : list of 4 tensors
            From ``SamProjector`` -- shapes ``[B*S, C_i, H_i, W_i]``.
        images : Tensor [B, S, 3, H, W]
            Original input images (used for shape inference).
        patch_start_idx : int
            (Unused -- kept for API symmetry with other heads.)
        point_feature : list of 3 tensors, optional
            Intermediate DPT features from the point head.

        Returns
        -------
        Tensor [B, S, output_dim, H, W]
        """
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = self._scratch_forward(multi_scale_features, point_feat=point_feature)

        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.window_self_atten(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        target_h = int(patch_h * self.patch_size)
        target_w = int(patch_w * self.patch_size)
        out = _custom_interpolate(out, (target_h, target_w), mode="bilinear", align_corners=True)

        out = self.output_conv2(out)

        return out.view(B, S, *out.shape[1:])

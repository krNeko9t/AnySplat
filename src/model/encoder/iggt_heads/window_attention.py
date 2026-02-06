"""Swin-style windowed self-attention (SwinSA) and cross-attention (SwinCA).

Adapted from IGGT/iggt/heads/window_sa.py – removed basicsr dependency;
utility helpers (to_2tuple, trunc_normal_) are inlined.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .attention_blocks import MemEffAttention

# ---------------------------------------------------------------------------
# Helpers (replace basicsr.archs.arch_util)
# ---------------------------------------------------------------------------


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02, a: float = -2.0, b: float = 2.0):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B = x.shape[0]
        return x.transpose(1, 2).contiguous().view(B, self.embed_dim, x_size[0], x_size[1])


# ---------------------------------------------------------------------------
# HAB  –  Hybrid Attention Block (self-attention)
# ---------------------------------------------------------------------------


class HAB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = MemEffAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None  # noqa: F841 – kept for clarity

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# OCAB  –  Overlapping Cross-Attention Block
# ---------------------------------------------------------------------------


class OCAB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        norm_layer=nn.LayerNorm,
        **_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )
        _trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, k_in, v_in, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        k_in = self.norm1(k_in).view(b, h, w, c)
        v_in = self.norm1(v_in).view(b, h, w, c)

        q = self.q(x).permute(0, 3, 1, 2)
        k_proj = self.k(k_in).permute(0, 3, 1, 2)
        v_proj = self.v(v_in).permute(0, 3, 1, 2)
        kv = torch.cat((k_proj, v_proj), dim=1)

        q_windows = window_partition(q.permute(0, 2, 3, 1), self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)

        kv_windows = self.unfold(kv)
        kv_windows = rearrange(
            kv_windows,
            "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        ).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q_h = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k_h = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v_h = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        q_h = q_h * self.scale
        attn = q_h @ k_h.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn_windows = (attn @ v_h).transpose(1, 2).reshape(b_, nq, self.dim)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)
        x = x.view(b, h * w, self.dim)
        x = self.proj(x) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SwinSA  –  Swin-style windowed self-attention module
# ---------------------------------------------------------------------------


class SwinSA(nn.Module):
    def __init__(
        self,
        img_size: int = 320,
        out_chans: int = 128,
        embed_dim: int = 128,
        num_heads: int = 6,
        window_size: int = 16,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        resi_connection: str = "1conv",
        **_kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.img_range = 1.0
        self.mean = torch.zeros(1, 1, 1, 1)

        relative_position_index_SA = self._calculate_rpi_sa()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.patch_unembed = PatchUnEmbed(patch_size=1, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.atten_block = HAB(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Identity()

        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self._calculate_mask(x_size).to(x.device)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.atten_block(x, x_size, self.relative_position_index_SA, attn_mask)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, H, W, C), Output: (B, H, W, C)."""
        x = x.permute(0, 3, 1, 2)
        x = self.conv_after_body(self._forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(x)
        return x.permute(0, 2, 3, 1).contiguous()


# ---------------------------------------------------------------------------
# SwinCA  –  Swin-style windowed cross-attention module
# ---------------------------------------------------------------------------


class SwinCA(nn.Module):
    def __init__(
        self,
        img_size: int = 320,
        out_chans: int = 128,
        embed_dim: int = 128,
        num_heads: int = 6,
        window_size: int = 16,
        overlap_ratio: float = 0.5,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        resi_connection: str = "1conv",
        **_kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        relative_position_index_SA = self._calculate_rpi_sa()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.patch_unembed = PatchUnEmbed(patch_size=1, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.atten_block = OCAB(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Identity()

        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)

        relative_position_index_OCA = self._calculate_rpi_oca()
        self.register_buffer("relative_position_index_OCA", relative_position_index_OCA)

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _calculate_rpi_oca(self):
        ws_ori = self.window_size
        ws_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_ori = torch.stack(torch.meshgrid(
            [torch.arange(ws_ori), torch.arange(ws_ori)], indexing="ij"
        ))
        coords_ext = torch.stack(torch.meshgrid(
            [torch.arange(ws_ext), torch.arange(ws_ext)], indexing="ij"
        ))
        coords_ori_flat = torch.flatten(coords_ori, 1)
        coords_ext_flat = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flat[:, None, :] - coords_ori_flat[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws_ori - ws_ext + 1
        relative_coords[:, :, 1] += ws_ori - ws_ext + 1
        relative_coords[:, :, 0] *= ws_ori + ws_ext - 1
        return relative_coords.sum(-1)

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _forward_features(self, x, k, v):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        k = self.patch_embed(k)
        v = self.patch_embed(v)
        x = self.pos_drop(x)
        k = self.pos_drop(k)
        v = self.pos_drop(v)
        x = self.atten_block(x, k, v, x_size, self.relative_position_index_OCA)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Input: (B, H, W, C), Output: (B, H, W, C)."""
        x = x.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        x = self.conv_after_body(self._forward_features(x, k, v)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(x)
        return x.permute(0, 2, 3, 1).contiguous()

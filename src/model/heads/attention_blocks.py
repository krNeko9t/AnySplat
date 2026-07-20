"""Attention building blocks for PartHead.

Adapted from IGGT/iggt/heads/block.py – xformers dependency is optional;
falls back to plain scaled-dot-product attention when unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    import xformers.ops
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class MemEffAttention(nn.Module):
    """Multi-head self-attention with optional xformers memory-efficient impl."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x: Tensor, xpos: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            qkv = qkv.permute(2, 0, 1, 3, 4)  # 3, B, N, H, D
            q, k, v = qkv[0], qkv[1], qkv[2]
            if self.qk_norm:
                q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())
            x = memory_efficient_attention(q, k, v, attn_bias=None).reshape(B, N, C)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
            q, k, v = qkv[0], qkv[1], qkv[2]
            if self.qk_norm:
                q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class MemEffCrossAttention(nn.Module):
    """Multi-head cross-attention with optional xformers memory-efficient impl."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        qpos: Tensor | None = None,
        kpos: Tensor | None = None,
    ) -> Tensor:
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads)

        if self.qk_norm:
            q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())

        if XFORMERS_AVAILABLE:
            x = memory_efficient_attention(q, k, v, scale=self.scale).reshape(B, Nq, C)
        else:
            q = q.permute(0, 2, 1, 3)  # B, H, Nq, D
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

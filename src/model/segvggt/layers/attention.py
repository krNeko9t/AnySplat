# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False
_ATTN_QK_DUMP_COUNTER = {}


def _dump_cross_attention_qk(q: Tensor, k: Tensor, layer_id: int, tokens_per_frame: int) -> None:
    """
    Temporarily dump normalized cross-attention q/k tensors for offline visualization.

    Files are organized as:
        paper_vis/attention_qk/layer_{layer_id:03d}/call_{idx:04d}.pt
    """
    layer_name = "unknown" if layer_id is None else f"{int(layer_id):03d}"
    save_dir = os.path.join("paper_vis", "attention_qk", f"layer_{layer_name}")
    os.makedirs(save_dir, exist_ok=True)

    call_idx = _ATTN_QK_DUMP_COUNTER.get(layer_name, 0)
    _ATTN_QK_DUMP_COUNTER[layer_name] = call_idx + 1

    payload = {
        "layer_id": -1 if layer_id is None else int(layer_id),
        "call_idx": int(call_idx),
        "tokens_per_frame": int(tokens_per_frame),
        "num_frames": int(k.shape[2] // tokens_per_frame) if tokens_per_frame > 0 else -1,
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
    }
    save_path = os.path.join(save_dir, f"call_{call_idx:04d}.pt")
    torch.save(payload, save_path)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply position encoding
        if self.rope is not None:
            # Use RoPE for 2D spatial positions
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        elif pos is not None:
            # Use additive position encoding when rope is None but pos is provided
            # pos should have shape (B, N, head_dim) or will be broadcasted
            # Reshape pos to match q, k: (B, num_heads, N, head_dim)
            if pos.dim() == 3:  # (B, N, C)
                # Reshape to (B, N, num_heads, head_dim)
                pos_reshaped = pos.reshape(B, N, self.num_heads, self.head_dim)
                # Permute to (B, num_heads, N, head_dim)
                pos_reshaped = pos_reshaped.permute(0, 2, 1, 3)
                q = q + pos_reshaped
                k = k + pos_reshaped
            else:
                # Assume pos is already in the right shape or can be broadcasted
                q = q + pos
                k = k + pos

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention module where query comes from one tensor and key/value come from another tensor.
    
    Args:
        dim: Output dimension (and default input dimension for q/k/v if not specified separately)
        dim_q: Input dimension for query. If None, uses `dim`.
        dim_kv: Input dimension for key/value. If None, uses `dim`.
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in q/k/v projection layers
        proj_bias: Whether to use bias in output projection layer
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        norm_layer: Normalization layer class
        qk_norm: Whether to apply normalization to q and k
        fused_attn: Whether to use fused attention implementation
    """
    def __init__(
        self,
        dim: int,
        dim_q: int = None,
        dim_kv: int = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        # Use dim as default for dim_q and dim_kv if not specified
        self.dim_q = dim_q if dim_q is not None else dim
        self.dim_kv = dim_kv if dim_kv is not None else dim
        self.dim = dim  # Output dimension
        
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # Separate linear layers for query, key, and value with independent input dimensions
        self.q_proj = nn.Linear(self.dim_q, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.dim_kv, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.dim_kv, dim, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, self.dim_q, bias=proj_bias)  # Project back to query dimension
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        P: int,
        supervise_attend_frame: bool = False,
        attn_mask: Tensor = None,
        layer_id: int = None,
    ) -> Tensor:
        """
        Args:
            x_q: Query tensor of shape (B, N_q, dim_q)
            x_kv: Key/Value tensor of shape (B, N_kv, dim_kv)
            P: int - number of tokens per frame
            supervise_attend_frame: bool - whether to supervise attention to frames
            attn_mask: Attention mask of shape (B, N_q, N_kv), True means masked (not attended to)

        Returns:
            Tuple(Tensor, Tensor): 
                - Output tensor of shape (B, N_q, dim_q)
                - Frame-level attention mass tensor of shape (B, N_q, num_frames), or None if supervise_attend_frame is False
        """
        B, N_q, C_q = x_q.shape
        B_kv, N_kv, C_kv = x_kv.shape
        num_frames = N_kv // P
        assert B == B_kv, "Batch size must match"
        assert C_q == self.dim_q, f"Query dimension mismatch: expected {self.dim_q}, got {C_q}"
        assert C_kv == self.dim_kv, f"Key/Value dimension mismatch: expected {self.dim_kv}, got {C_kv}"
        
        # Project to get query, key, value (all project to self.dim)
        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply normalization
        q, k = self.q_norm(q), self.k_norm(k)
        # _dump_cross_attention_qk(q, k, layer_id=layer_id, tokens_per_frame=P)

        # Convert boolean mask to attention bias for scaled_dot_product_attention
        # attn_mask: (B, N_q, N_kv), True means masked -> need to convert to float mask with -inf for masked positions
        if attn_mask is not None:
            # Expand mask for multi-head attention: (B, N_q, N_kv) -> (B, num_heads, N_q, N_kv)
            attn_bias = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # Convert boolean mask to float: True (masked) -> -inf, False (not masked) -> 0
            attn_bias = torch.zeros_like(attn_bias, dtype=q.dtype).masked_fill(attn_bias, float('-inf'))
        else:
            attn_bias = None

        if self.fused_attn and (not supervise_attend_frame):
            x = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
            attn_frame_mean = None
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # (B, num_heads, N_q, N_kv)
            
            # Apply attention mask before softmax
            if attn_bias is not None:
                attn = attn + attn_bias
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

            if supervise_attend_frame:
                # average over heads
                attn_mean = attn.mean(dim=1)  # (B, N_q, N_kv)
                # reshape to (B, N_q, num_frames, P)
                attn_mean = attn_mean.view(B, N_q, num_frames, P)
                # compute mean over tokens in each frame
                attn_frame_mean = attn_mean.mean(dim=-1)  # (B, N_q, num_frames)
            else:
                attn_frame_mean = None

        x = x.transpose(1, 2).reshape(B, N_q, self.dim)
        x = self.proj(x)  # Project back to dim_q
        x = self.proj_drop(x)
        
        return x, attn_frame_mean


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

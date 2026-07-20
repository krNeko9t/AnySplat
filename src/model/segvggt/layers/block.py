# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, CrossAttention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


class CrossBlock(nn.Module):
    """
    Cross-attention block: performs attention where query comes from a separate tensor
    (e.g., instance queries) and key/value come from another tensor (e.g., tokens).

    Follows the residual/norm/ffn pattern of Block to keep behavior consistent.
    
    Args:
        dim: Output dimension (and default input dimension for q/k/v if not specified separately)
        num_heads: Number of attention heads
        dim_q: Input dimension for query. If None, uses `dim`.
        dim_kv: Input dimension for key/value. If None, uses `dim`.
        mlp_ratio: Ratio of MLP hidden dimension to dim_q
        qkv_bias: Whether to use bias in q/k/v projection layers
        proj_bias: Whether to use bias in output projection layer
        ffn_bias: Whether to use bias in MLP layers
        drop: Dropout rate
        attn_drop: Dropout rate for attention weights
        init_values: Initial value for LayerScale
        drop_path: Stochastic depth rate
        act_layer: Activation layer class
        norm_layer: Normalization layer class
        qk_norm: Whether to apply normalization to q and k
        fused_attn: Whether to use fused attention implementation
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_q: int = None,
        dim_kv: int = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
    ) -> None:
        super().__init__()

        # Use dim as default for dim_q and dim_kv if not specified
        self.dim_q = dim_q if dim_q is not None else dim
        self.dim_kv = dim_kv if dim_kv is not None else dim

        # Norm for queries and for key/value (use appropriate dimensions)
        self.norm_q = norm_layer(self.dim_q)
        self.norm_kv = norm_layer(self.dim_kv)

        # Use our custom CrossAttention module with independent dimensions
        self.attn = CrossAttention(
            dim=dim,
            dim_q=self.dim_q,
            dim_kv=self.dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )

        self.ls1 = LayerScale(self.dim_q, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(self.dim_q)
        mlp_hidden_dim = int(self.dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim_q, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(self.dim_q, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        P: int,
        supervise_attend_frame: bool,
        attn_mask: Tensor = None,
        layer_id: int = None,
    ) -> Tensor:
        """
        x_q: (B, Nq, C) - queries that will be updated
        x_kv: (B, Nk, C) - keys/values to attend to
        P: int - number of tokens per frame
        supervise_attend_frame: bool - whether to supervise attention to frames
        attn_mask: (B, Nq, Nk) - attention mask, True means masked (not attended to)

        returns: updated x_q of shape (B, Nq, C)
        """

        def attn_residual_func(q, kv, P, supervise_attend_frame, attn_mask, layer_id):
            # Apply norms and cross-attention
            qn = self.norm_q(q)
            kvn = self.norm_kv(kv)
            attn_out, attn_frame_mean = self.attn(qn, kvn, P, supervise_attend_frame, attn_mask, layer_id=layer_id)
            return self.ls1(attn_out), attn_frame_mean

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        # Apply cross-attention with residual connection
        if self.training and self.sample_drop_ratio > 0.0:
            # For simplicity, use standard dropout path for cross-attention
            attn_out, attn_frame_mean = attn_residual_func(x_q, x_kv, P, supervise_attend_frame, attn_mask, layer_id)
            x_q = x_q + self.drop_path1(attn_out)
            x_q = x_q + self.drop_path2(ffn_residual_func(x_q))
        else:
            attn_out, attn_frame_mean = attn_residual_func(x_q, x_kv, P, supervise_attend_frame, attn_mask, layer_id)
            x_q = x_q + attn_out
            x_q = x_q + ffn_residual_func(x_q)

        return x_q, attn_frame_mean


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

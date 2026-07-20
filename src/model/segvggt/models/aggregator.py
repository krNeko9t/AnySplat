# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from VGGT (https://github.com/facebookresearch/vggt).
#
# SegVGGT
# url: https://github.com/IDEA-Research/SegVGGT
# Copyright (c) 2026 IDEA. All Rights Reserved.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
from omegaconf import ListConfig

from src.model.segvggt.layers import PatchEmbed
from src.model.segvggt.layers.block import Block, CrossBlock
from src.model.segvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from src.model.segvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from src.model.segvggt.layers.lora import (
    apply_lora_to_attention,
    apply_lora_to_mlp,
    print_lora_statistics
)
logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer, with SegVGGT
    instance query extensions.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        enable_instance_seg (bool): Whether to enable instance segmentation.
        instance_query_num (int): Number of instance query tokens.
        instance_query_dim (int or None): Dimension of instance query tokens. If None, uses embed_dim.
        cross_block_layers (list[int] or None): List of global block indices to apply cross-attention for instance queries.
        cross_block_params (dict or None): Using custom parameters for cross-attention blocks instead of inheriting from global blocks.
            - mask_attention (bool): Whether to mask attention to invalid patches.
            - mask_threshold (float): Threshold to generate attention mask for cross-attention blocks.
        use_lora (bool): Whether to use LoRA for efficient fine-tuning. Default: False.
        lora_config (dict or None): LoRA configuration dict with keys:
            - rank (int): LoRA rank for attention layers, default 8
            - alpha (float): LoRA alpha for attention layers, default 8.0
            - dropout (float): LoRA dropout, default 0.0
            - mlp_rank (int or None): LoRA rank for MLP layers. If None, uses same as `rank`
            - mlp_alpha (float or None): LoRA alpha for MLP layers. If None, uses same as `alpha`
            - target_frame_blocks (bool): Apply LoRA to frame_blocks, default True
            - target_global_blocks (bool): Apply LoRA to global_blocks, default True
            - target_qkv (bool): Apply to QKV projection in attention, default True
            - target_proj (bool): Apply to output projection in attention, default True
            - target_mlp (bool): Apply to MLP layers (fc1 and fc2), default False
            - target_fc1 (bool): Apply to MLP fc1 layer, default True (only if target_mlp=True)
            - target_fc2 (bool): Apply to MLP fc2 layer, default True (only if target_mlp=True)
            - verbose (bool): Print LoRA application details, default False
        query_self_attention (bool): Whether to use self-attention among instance queries.
        query_sa_layers (list[int] or None): List of layers to apply instance query self-attention. If None, applies to all layers.
        query_sa_params (dict or None): Using custom parameters for instance query self-attention blocks.
        supervise_attend_frame (bool): Whether to supervise cross-attention between frames for instance segmentation.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_instance_seg=False,
        instance_query_num=400,
        instance_query_dim=None,
        cross_block_layers=None,
        cross_block_params=None,
        use_lora=False,
        lora_config=None,
        query_self_attention=False,
        query_sa_layers=None,
        query_sa_params=None,
        supervise_attend_frame=False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # Add more register token as queries for instance segmentation
        self.enable_instance_seg = enable_instance_seg
        if self.enable_instance_seg:
            self.instance_query_num = instance_query_num
            self.instance_query_dim = instance_query_dim if instance_query_dim is not None else embed_dim
            self.instance_query_token = nn.Parameter(torch.randn(1, instance_query_num, self.instance_query_dim))
            nn.init.normal_(self.instance_query_token, std=1e-6)

            self.cross_block_layers = cross_block_layers
            if self.cross_block_layers is not None:
                assert isinstance(self.cross_block_layers, (list, ListConfig)), "cross_block_layers should be a list of integers."
            else:
                self.cross_block_layers = list(range(depth))
            # create a cross-attention block per global block to update instance queries
            self.cross_block_params = cross_block_params
            if self.cross_block_params is None:
                self.cross_block_params = {}
            self.instance_cross_blocks = nn.ModuleList(
                [
                    CrossBlock(
                        dim=self.cross_block_params.get("embed_dim", embed_dim),
                        num_heads=self.cross_block_params.get("num_heads", num_heads),
                        dim_q=self.instance_query_dim,
                        dim_kv=embed_dim,
                        mlp_ratio=self.cross_block_params.get("mlp_ratio", mlp_ratio),
                        qkv_bias=self.cross_block_params.get("qkv_bias", qkv_bias),
                        proj_bias=self.cross_block_params.get("proj_bias", proj_bias),
                        ffn_bias=self.cross_block_params.get("ffn_bias", ffn_bias),
                        init_values=self.cross_block_params.get("init_values", init_values),
                        qk_norm=self.cross_block_params.get("qk_norm", qk_norm),
                    )
                    if i in self.cross_block_layers
                    else None for i in range(depth)
                ]
            )
            # projection layer for instance queries to dot-product with feature maps
            self.instance_queries_proj = nn.Linear(self.instance_query_dim, 128)

            self.supervise_attend_frame = supervise_attend_frame

            # mask attention parameters (similar to Mask2Former)
            self.mask_attention = self.cross_block_params.get("mask_attention", False)
            self.mask_threshold = self.cross_block_params.get("mask_threshold", 0.4)

            # for query self-attention
            self.query_self_attention = query_self_attention
            self.query_sa_params = query_sa_params
            if query_sa_layers is not None:
                self.query_sa_layers = query_sa_layers
            else:
                self.query_sa_layers = list(range(depth))
            if self.query_self_attention:
                self.instance_query_self_attn = nn.ModuleList(
                    [
                        block_fn(
                            dim=self.instance_query_dim,
                            num_heads=self.query_sa_params.get("num_heads", num_heads),
                            mlp_ratio=self.query_sa_params.get("mlp_ratio", mlp_ratio),
                            qkv_bias=self.query_sa_params.get("qkv_bias", qkv_bias),
                            proj_bias=self.query_sa_params.get("proj_bias", proj_bias),
                            ffn_bias=self.query_sa_params.get("ffn_bias", ffn_bias),
                            init_values=self.query_sa_params.get("init_values", init_values),
                            qk_norm=self.query_sa_params.get("qk_norm", qk_norm),
                            rope=None,
                        )
                        if i in self.query_sa_layers else None for i in range(depth)
                    ]
                )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config or {})

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def _apply_lora(self, lora_config: dict):
        """
        Apply LoRA to frame_blocks, global_blocks, and optionally cross_blocks.
        
        Args:
            lora_config: Dict with LoRA configuration
                - rank (int): LoRA rank for attention layers, default 8
                - alpha (float): LoRA alpha for attention layers, default 8.0
                - dropout (float): LoRA dropout, default 0.0
                - mlp_rank (int or None): LoRA rank for MLP layers. If None, uses same as `rank`
                - mlp_alpha (float or None): LoRA alpha for MLP layers. If None, uses same as `alpha`
                - target_frame_blocks (bool): Apply to frame_blocks, default True
                - target_global_blocks (bool): Apply to global_blocks, default True
                - target_qkv (bool): Apply to QKV projection in attention, default True
                - target_proj (bool): Apply to output projection in attention, default True
                - target_mlp (bool): Apply to MLP layers (fc1 and fc2), default False
                - target_fc1 (bool): Apply to MLP fc1 layer, default True
                - target_fc2 (bool): Apply to MLP fc2 layer, default True
                - verbose (bool): Print details, default False
        """
        # Get current rank for distributed training (only print on rank 0)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            current_rank = dist.get_rank()
        else:
            current_rank = 0  # Single GPU or non-distributed mode
        
        rank = lora_config.get('rank', 8)
        alpha = lora_config.get('alpha', 8.0)
        dropout = lora_config.get('dropout', 0.0)
        
        # MLP-specific parameters (fallback to attention parameters if not specified)
        mlp_rank = lora_config.get('mlp_rank', None)
        mlp_alpha = lora_config.get('mlp_alpha', None)
        if mlp_rank is None:
            mlp_rank = rank
        if mlp_alpha is None:
            mlp_alpha = alpha
        
        target_frame = lora_config.get('target_frame_blocks', True)
        target_global = lora_config.get('target_global_blocks', True)
        target_qkv = lora_config.get('target_qkv', True)
        target_proj = lora_config.get('target_proj', True)
        target_mlp = lora_config.get('target_mlp', False)
        target_fc1 = lora_config.get('target_fc1', True)
        target_fc2 = lora_config.get('target_fc2', True)
        verbose = lora_config.get('verbose', False)
        
        total_modified = 0
        
        # Apply LoRA to frame_blocks
        if target_frame:
            if verbose and current_rank == 0:
                print(f"\n=== Applying LoRA to frame_blocks ===")
                print(f"  Attention: rank={rank}, alpha={alpha}")
                if target_mlp:
                    print(f"  MLP: rank={mlp_rank}, alpha={mlp_alpha}")
            for i, block in enumerate(self.frame_blocks):
                # Apply to attention layers
                if hasattr(block, 'attn'):
                    num = apply_lora_to_attention(
                        block.attn,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        target_qkv=target_qkv,
                        target_proj=target_proj,
                        verbose=verbose and current_rank == 0
                    )
                    total_modified += num
                    if verbose and current_rank == 0 and num > 0:
                        print(f"  Frame block {i} attention: {num} layers modified")
                
                # Apply to MLP layers if requested
                if target_mlp and hasattr(block, 'mlp'):
                    num = apply_lora_to_mlp(
                        block.mlp,
                        rank=mlp_rank,
                        alpha=mlp_alpha,
                        dropout=dropout,
                        target_fc1=target_fc1,
                        target_fc2=target_fc2,
                        verbose=verbose and current_rank == 0
                    )
                    total_modified += num
                    if verbose and num > 0:
                        print(f"  Frame block {i} MLP: {num} layers modified")
        
        # Apply LoRA to global_blocks
        if target_global:
            if verbose and current_rank == 0:
                print(f"\n=== Applying LoRA to global_blocks ===")
                print(f"  Attention: rank={rank}, alpha={alpha}")
                if target_mlp:
                    print(f"  MLP: rank={mlp_rank}, alpha={mlp_alpha}")
            for i, block in enumerate(self.global_blocks):
                # Apply to attention layers
                if hasattr(block, 'attn'):
                    num = apply_lora_to_attention(
                        block.attn,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        target_qkv=target_qkv,
                        target_proj=target_proj,
                        verbose=verbose and current_rank == 0
                    )
                    total_modified += num
                    if verbose and current_rank == 0 and num > 0:
                        print(f"  Global block {i} attention: {num} layers modified")
                
                # Apply to MLP layers if requested
                if target_mlp and hasattr(block, 'mlp'):
                    num = apply_lora_to_mlp(
                        block.mlp,
                        rank=mlp_rank,
                        alpha=mlp_alpha,
                        dropout=dropout,
                        target_fc1=target_fc1,
                        target_fc2=target_fc2,
                        verbose=verbose and current_rank == 0
                    )
                    total_modified += num
                    if verbose and current_rank == 0 and num > 0:
                        print(f"  Global block {i} MLP: {num} layers modified")
        
        if current_rank == 0:  # Only print summary on rank 0
            print(f"\n{'='*80}")
            print(f"LoRA Application Summary:")
            print(f"  Total layers modified: {total_modified}")
            print(f"  Attention - Rank: {rank}, Alpha: {alpha}")
            if target_mlp:
                print(f"  MLP - Rank: {mlp_rank}, Alpha: {mlp_alpha}")
                print(f"  MLP layers enabled: fc1={target_fc1}, fc2={target_fc2}")
            print(f"  Dropout: {dropout}")
            print(f"{'='*80}\n")
            
            # Print detailed statistics
            print_lora_statistics(self)


    def forward(self, images: torch.Tensor) -> Union[Tuple[List[torch.Tensor], int], Tuple[List[torch.Tensor], int, torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            When enable_instance_seg is False:
                (list[torch.Tensor], int):
                    The list of outputs from the attention blocks,
                    and the patch_start_idx indicating where patch tokens begin.
            When enable_instance_seg is True:
                (list[torch.Tensor], int, torch.Tensor):
                    The list of outputs from the attention blocks,
                    the patch_start_idx indicating where patch tokens begin,
                    and the updated instance queries of shape (B, instance_query_num, C).
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        if self.enable_instance_seg:
            instance_queries = self.instance_query_token.expand(B, -1, -1)  # (B, instance_query_num, C)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        attn_frame_mean_list = []
        
        # Initialize attention mask for mask attention (None for the first layer)
        attn_mask = None

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )

                    # After performing global attention, update instance queries via cross-attention and self-attention
                    if self.enable_instance_seg:
                        cross_idx = global_idx - 1

                        if cross_idx in self.cross_block_layers:
                            instance_queries, attn_frame_mean, attn_mask = self._process_instance_query_cross_attention(
                                tokens, instance_queries, B, S, P, C, cross_idx, attn_mask
                            )
                            if self.supervise_attend_frame and attn_frame_mean is not None:
                                attn_frame_mean_list.append(attn_frame_mean)
                        if self.query_self_attention and cross_idx in self.query_sa_layers:
                            instance_queries = self._process_instance_query_self_attention(instance_queries, cross_idx)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        
        if self.enable_instance_seg:
            return output_list, self.patch_start_idx, instance_queries, attn_frame_mean_list
        else:
            return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
    
    def _process_instance_query_cross_attention(self, tokens, instance_queries, B, S, P, C, cross_idx, attn_mask=None, positional_queries_emb_ca=None):
        """
        Process instance query cross-attention blocks. We keep tokens in shape (B, S*P, C).
        Instance queries are in shape (B, Q, C), where Q is the number of instance queries.
        
        Args:
            tokens: Token features of shape (B*S, P, C) or (B, S*P, C)
            instance_queries: Instance query features of shape (B, Q, C_q)
            B: Batch size
            S: Number of frames
            P: Number of tokens per frame
            C: Token embedding dimension
            cross_idx: Index of the cross-attention block
            attn_mask: Attention mask of shape (B, Q, S*P), True means masked (not attended to)
            positional_queries_emb_ca: Positional embedding for instance queries of shape (B, Q, C_q), optional
        
        Returns:
            instance_queries: Updated instance queries
            attn_frame_mean: Frame-level attention weights (if supervise_attend_frame is True)
            next_attn_mask: Attention mask for the next layer (if mask_attention is True)
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        # Apply positional embedding to instance queries if provided
        if positional_queries_emb_ca is not None:
            instance_queries_with_pos = instance_queries + positional_queries_emb_ca
        else:
            instance_queries_with_pos = instance_queries

        if self.instance_cross_blocks[cross_idx].training:
            instance_queries, attn_frame_mean = checkpoint(
                self.instance_cross_blocks[cross_idx], 
                instance_queries_with_pos, tokens, P, self.supervise_attend_frame, attn_mask, cross_idx,
                use_reentrant=self.use_reentrant
            )
        else:
            instance_queries, attn_frame_mean = self.instance_cross_blocks[cross_idx](
                instance_queries_with_pos, tokens, P, self.supervise_attend_frame, attn_mask, cross_idx
            )

        # Compute attention mask for the next layer (similar to Mask2Former)
        next_attn_mask = None
        if self.mask_attention:
            # Reuse the cross-attention projections to avoid adding parameters.
            # The CrossBlock contains an `attn` module (CrossAttention) with q_proj/k_proj
            block = self.instance_cross_blocks[cross_idx]
            attn_module = block.attn

            # Compute mask only over patch tokens (exclude camera/register special tokens)
            # tokens: (B, S*P, C) -> (B, S, P, C)
            tokens_rs = tokens.view(B, S, P, C)
            # select patch tokens per frame from patch_start_idx onwards
            patch_start = self.patch_start_idx
            tokens_patches = tokens_rs[:, :, patch_start:, :].contiguous()  # (B, S, P_patch, C)
            Bp, Sp, P_patch, C_ = tokens_patches.shape
            # flatten patch tokens to (B, N_patch, C)
            tokens_patches_flat = tokens_patches.view(B, Sp * P_patch, C_)

            # Project queries and these patch-keys into shared attention space
            # q_for_mask: (B, Q, dim), k_for_mask: (B, N_patch, dim)
            q_for_mask = attn_module.q_proj(instance_queries)
            k_for_mask = attn_module.k_proj(tokens_patches_flat)

            # similarity over only patch tokens: (B, Q, N_patch)
            similarity_patch = torch.einsum('bqd,bnd->bqn', q_for_mask, k_for_mask)

            # Apply sigmoid and threshold to get binary mask for patch tokens; detach to stop gradients
            mask_pred_patch = torch.sigmoid(similarity_patch).detach()
            mask_patch = mask_pred_patch < self.mask_threshold  # (B, Q, N_patch)

            # If a query masks all patch tokens, unmask all patch tokens for that query
            all_masked_patch = mask_patch.all(dim=-1)  # (B, Q)
            if all_masked_patch.any():
                mask_patch = mask_patch.masked_fill(
                    all_masked_patch.unsqueeze(-1).expand_as(mask_patch), False
                )

            # Build full-length mask over all tokens (special tokens remain unmasked)
            # full_mask reshaped to (B, Q, S, P) to place mask_patch into [:, :, :, patch_start:]
            full_mask_rs = torch.zeros(B, instance_queries.shape[1], S, P, dtype=torch.bool, device=tokens.device)
            mask_patch_rs = mask_patch.view(B, instance_queries.shape[1], S, P_patch)
            full_mask_rs[:, :, :, patch_start:] = mask_patch_rs
            next_attn_mask = full_mask_rs.view(B, instance_queries.shape[1], S * P)

        return instance_queries, attn_frame_mean, next_attn_mask

    def _process_instance_query_self_attention(self, instance_queries, cross_idx, positional_queries_emb_sa=None):
        """
        Process instance query self-attention blocks.
        Instance queries are in shape (B, Q, C), where Q is the number of instance queries.
        
        Args:
            instance_queries: Instance query features of shape (B, Q, C)
            cross_idx: Index of the self-attention block
            positional_queries_emb_sa: Positional embedding for instance queries of shape (B, Q, C), optional
        
        Returns:
            instance_queries: Updated instance queries of shape (B, Q, C)
        """
        # Apply positional embedding to instance queries if provided
        if positional_queries_emb_sa is not None:
            pos = positional_queries_emb_sa
        else:
            pos = None
        
        if self.instance_query_self_attn[cross_idx].training:
            instance_queries = checkpoint(self.instance_query_self_attn[cross_idx], instance_queries, pos, use_reentrant=self.use_reentrant)
        else:
            instance_queries = self.instance_query_self_attn[cross_idx](instance_queries, pos=pos)

        return instance_queries


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

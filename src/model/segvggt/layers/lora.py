# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Low-Rank Adaptation (LoRA) implementation for efficient fine-tuning.

Reference:
    LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank decomposition.
    
    Implements: h = W0*x + (alpha/r) * B*A*x
    where W0 is frozen, A is (r, in_features), B is (out_features, r)
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: Rank of the low-rank decomposition (r)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout rate for LoRA path
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices using nn.Linear
        # A: (in_features -> rank) - initialized with Kaiming uniform
        # B: (rank -> out_features) - initialized with zeros
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters following the paper."""
        # Initialize A with Kaiming uniform (same as default nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Initialize B with zeros (so initially LoRA has no effect)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA forward pass: (alpha/r) * B @ A @ x
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply dropout before LoRA computation
        x_dropout = self.lora_dropout(x)
        
        # Compute: B(A(x))
        # Step 1: A(x): (..., in_features) -> (..., rank)
        lora_out = self.lora_A(x_dropout)
        # Step 2: B(..., rank) -> (..., out_features)
        lora_out = self.lora_B(lora_out)
        
        # Scale by alpha/rank
        lora_out = lora_out * self.scaling
        
        return lora_out


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This module wraps an existing nn.Linear layer and adds LoRA parameters.
    The original linear layer's parameters are frozen.
    
    Args:
        linear: The original nn.Linear layer to wrap
        rank: Rank of LoRA decomposition
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
    """
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = linear(x) + lora(x)
        
        Args:
            x: Input tensor
        
        Returns:
            Output with LoRA adaptation applied
        """
        return self.linear(x) + self.lora(x)
    
    @property
    def weight(self):
        """Expose weight for compatibility."""
        return self.linear.weight
    
    @property
    def bias(self):
        """Expose bias for compatibility."""
        return self.linear.bias


def apply_lora_to_linear(
    module: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    verbose: bool = False,
):
    """
    Recursively apply LoRA to all nn.Linear layers matching target names.
    
    Args:
        module: The module to apply LoRA to
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'v_proj'])
                       If None, applies to all nn.Linear layers
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout rate
        verbose: Whether to print which modules are modified
    
    Returns:
        Number of layers modified
    """
    num_modified = 0
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if this layer should get LoRA
            should_apply = (target_modules is None or 
                          any(target in name for target in target_modules))
            
            if should_apply:
                # Replace with LoRA-wrapped version
                lora_linear = LinearWithLoRA(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_linear)
                num_modified += 1
                if verbose:
                    print(f"Applied LoRA to: {name} (in_features={child.in_features}, "
                          f"out_features={child.out_features}, rank={rank})")
        else:
            # Recursively apply to submodules
            num_modified += apply_lora_to_linear(
                child, target_modules, rank, alpha, dropout, verbose
            )
    
    return num_modified


def apply_lora_to_attention(
    attention_module: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_qkv: bool = True,
    target_proj: bool = True,
    verbose: bool = False,
):
    """
    Apply LoRA specifically to attention layers.
    
    For standard Attention module, this targets:
    - qkv projection (if target_qkv=True)
    - output projection (if target_proj=True)
    
    Args:
        attention_module: The Attention module
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_qkv: Whether to apply LoRA to QKV projection
        target_proj: Whether to apply LoRA to output projection
        verbose: Whether to print modifications
    
    Returns:
        Number of layers modified
    """
    num_modified = 0
    
    # Target QKV projection
    if target_qkv and hasattr(attention_module, 'qkv'):
        qkv_layer = attention_module.qkv
        if isinstance(qkv_layer, nn.Linear):
            lora_qkv = LinearWithLoRA(qkv_layer, rank=rank, alpha=alpha, dropout=dropout)
            attention_module.qkv = lora_qkv
            num_modified += 1
            if verbose:
                print(f"Applied LoRA to attention.qkv "
                      f"(in={qkv_layer.in_features}, out={qkv_layer.out_features}, rank={rank})")
    
    # Target output projection
    if target_proj and hasattr(attention_module, 'proj'):
        proj_layer = attention_module.proj
        if isinstance(proj_layer, nn.Linear):
            lora_proj = LinearWithLoRA(proj_layer, rank=rank, alpha=alpha, dropout=dropout)
            attention_module.proj = lora_proj
            num_modified += 1
            if verbose:
                print(f"Applied LoRA to attention.proj "
                      f"(in={proj_layer.in_features}, out={proj_layer.out_features}, rank={rank})")
    
    return num_modified


def apply_lora_to_mlp(
    mlp_module: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_fc1: bool = True,
    target_fc2: bool = True,
    verbose: bool = False,
):
    """
    Apply LoRA specifically to MLP layers.
    
    For standard MLP module, this targets:
    - fc1 (first linear layer, if target_fc1=True)
    - fc2 (second linear layer, if target_fc2=True)
    
    Args:
        mlp_module: The MLP module
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_fc1: Whether to apply LoRA to fc1
        target_fc2: Whether to apply LoRA to fc2
        verbose: Whether to print modifications
    
    Returns:
        Number of layers modified
    """
    num_modified = 0
    
    # Target fc1 (first MLP layer)
    if target_fc1 and hasattr(mlp_module, 'fc1'):
        fc1_layer = mlp_module.fc1
        if isinstance(fc1_layer, nn.Linear):
            lora_fc1 = LinearWithLoRA(fc1_layer, rank=rank, alpha=alpha, dropout=dropout)
            mlp_module.fc1 = lora_fc1
            num_modified += 1
            if verbose:
                print(f"Applied LoRA to mlp.fc1 "
                      f"(in={fc1_layer.in_features}, out={fc1_layer.out_features}, rank={rank})")
    
    # Target fc2 (second MLP layer)
    if target_fc2 and hasattr(mlp_module, 'fc2'):
        fc2_layer = mlp_module.fc2
        if isinstance(fc2_layer, nn.Linear):
            lora_fc2 = LinearWithLoRA(fc2_layer, rank=rank, alpha=alpha, dropout=dropout)
            mlp_module.fc2 = lora_fc2
            num_modified += 1
            if verbose:
                print(f"Applied LoRA to mlp.fc2 "
                      f"(in={fc2_layer.in_features}, out={fc2_layer.out_features}, rank={rank})")
    
    return num_modified


def get_lora_parameters(model: nn.Module):
    """
    Get all LoRA parameters from a model.
    
    Args:
        model: The model containing LoRA layers
    
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            # Get parameters from the Linear layers
            lora_params.extend(list(module.lora_A.parameters()))
            lora_params.extend(list(module.lora_B.parameters()))
    return lora_params


def count_lora_parameters(model: nn.Module):
    """
    Count total number of LoRA parameters.
    
    Args:
        model: The model containing LoRA layers
    
    Returns:
        Total number of trainable LoRA parameters
    """
    return sum(p.numel() for p in get_lora_parameters(model))


def print_lora_statistics(model: nn.Module):
    """
    Print statistics about LoRA layers in the model.
    
    Args:
        model: The model containing LoRA layers
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = count_lora_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 80)
    print("LoRA Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA percentage: {100 * lora_params / total_params:.2f}%")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print("=" * 80)

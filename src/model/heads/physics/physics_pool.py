"""Shared instance-masked average pooling for physics readouts."""

from __future__ import annotations

import torch
from torch import Tensor


def pool_instance_features(
    feat_map: Tensor,
    instance_mask: Tensor,
    valid_mask: Tensor | None = None,
    ignore_id: int = 0,
) -> tuple[list[Tensor], list[Tensor]]:
    """Masked average-pool dense features per instance.

    Parameters
    ----------
    feat_map : [B, V, C, H, W]
    instance_mask : [B, V, H, W]
    valid_mask : optional instance-*quality* mask [B, V, H, W] (or [..., 1]).
        False pixels are dropped from pooling.  Must NOT be depth ``valid_mask``
        (depth holes ≠ untrustworthy instance labels).
    ignore_id : instance id to exclude (default 0)

    Returns
    -------
    pooled : list length B, each [K_b, C]
    ids : list length B, each [K_b]
    """
    B = feat_map.shape[0]
    pooled_list: list[Tensor] = []
    ids_list: list[Tensor] = []
    for b in range(B):
        vm_b = valid_mask[b] if valid_mask is not None else None
        pooled, ids = pool_one_sample(
            feat_map[b], instance_mask[b], vm_b, ignore_id=ignore_id
        )
        pooled_list.append(pooled)
        ids_list.append(ids)
    return pooled_list, ids_list


def pool_one_sample(
    feat_map: Tensor,
    inst_mask: Tensor,
    valid_mask: Tensor | None,
    ignore_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Pool one sample. feat_map [V,C,H,W], mask [V,H,W] → ([K,C], [K])."""
    V, C, H, W = feat_map.shape

    if valid_mask is not None:
        if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        inst_mask = inst_mask.clone()
        inst_mask[~valid_mask.bool()] = ignore_id

    flat_mask = inst_mask.reshape(-1)
    unique_ids = torch.unique(flat_mask)
    unique_ids = unique_ids[unique_ids != ignore_id]

    if unique_ids.numel() == 0:
        return (
            torch.empty((0, C), device=feat_map.device, dtype=feat_map.dtype),
            torch.empty((0,), device=feat_map.device, dtype=torch.long),
        )

    feat_flat = feat_map.reshape(V, C, -1)
    mask_flat = inst_mask.reshape(V, -1)

    pooled_rows: list[Tensor] = []
    id_rows: list[Tensor] = []
    for inst_id in unique_ids:
        per_view = mask_flat == inst_id
        count = per_view.sum()
        if count == 0:
            continue
        masked = feat_flat * per_view.unsqueeze(1).to(feat_flat.dtype)
        pooled = masked.sum(dim=(0, 2)) / count.clamp(min=1).to(feat_flat.dtype)
        pooled_rows.append(pooled)
        id_rows.append(inst_id)

    if not pooled_rows:
        return (
            torch.empty((0, C), device=feat_map.device, dtype=feat_map.dtype),
            torch.empty((0,), device=feat_map.device, dtype=torch.long),
        )

    return torch.stack(pooled_rows, dim=0), torch.stack(id_rows, dim=0)

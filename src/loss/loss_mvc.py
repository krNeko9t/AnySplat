from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.instseg.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMvcCfg:
    weight: float = 1.0
    num_samples: int = 8192
    margin: float = 1.0
    lambda_pull: float = 2.0
    lambda_push: float = 1.0
    ignore_id: int = 0
    block_size: int = 512
    # If enabled, divide by number of evaluated pairs (ordered, excluding diagonal).
    normalize_by_pairs: bool = False


@dataclass
class LossMvcCfgWrapper:
    mvc: LossMvcCfg


class LossMVC(Loss[LossMvcCfg, LossMvcCfgWrapper]):
    """3D-consistent multi-view contrastive loss on per-pixel instance embeddings.

    Expected inputs (provided via `depth_dict` by the training loop):
    - depth_dict['instance_feat_map']: Float[Tensor] shaped [B, V, N, H, W]
    - depth_dict['instance_mask']: Int64[Tensor] shaped [B, V, H, W]
    - optional depth_dict['instance_valid_mask']: Bool[Tensor] shaped [B, V, H, W]
    """

    def __init__(self, cfg: LossMvcCfgWrapper) -> None:
        super().__init__(cfg)
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if depth_dict is None:
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        feat_map: Tensor | None = depth_dict.get("instance_feat_map")
        inst_mask: Tensor | None = depth_dict.get("instance_mask")
        valid_mask: Tensor | None = depth_dict.get("instance_valid_mask")

        if feat_map is None or inst_mask is None:
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        # [B, V, N, H, W] / [B, V, H, W]
        B, V, N, H, W = feat_map.shape
        device = feat_map.device

        total_pull = torch.tensor(0.0, device=device)
        total_push = torch.tensor(0.0, device=device)
        total_pairs = torch.tensor(0.0, device=device)

        for b in range(B):
            feats_b = feat_map[b]  # [V, N, H, W]
            ids_b = inst_mask[b]  # [V, H, W]
            if valid_mask is None:
                valid_b = torch.ones_like(ids_b, dtype=torch.bool, device=device)
            else:
                valid_b = valid_mask[b].to(torch.bool)

            eligible = valid_b & (ids_b != int(self.cfg.ignore_id))
            if eligible.sum() < 2:
                continue

            # Flatten (V,H,W) -> P.
            ids_flat = ids_b.reshape(-1)
            elig_flat = eligible.reshape(-1)

            idx_all = torch.nonzero(elig_flat, as_tuple=False).squeeze(1)
            S = min(int(self.cfg.num_samples), int(idx_all.numel()))
            if S < 2:
                continue

            # Sample without replacement.
            perm = torch.randperm(idx_all.numel(), device=device)[:S]
            idx = idx_all[perm]  # [S]

            # Gather features: [V,N,H,W] -> [P,N]
            feats_flat = feats_b.permute(0, 2, 3, 1).reshape(-1, N)  # [P, N]
            f = feats_flat[idx]  # [S, N]

            # L2 normalize before distance.
            f = F.normalize(f, p=2, dim=-1, eps=1e-8)
            ids = ids_flat[idx].to(torch.int64)  # [S]

            # Precompute indices for diagonal masking.
            point_ids = torch.arange(S, device=device)

            pull = torch.tensor(0.0, device=device)
            push = torch.tensor(0.0, device=device)

            # Full pairwise (ordered pairs, excluding diagonal), computed in blocks.
            bs = max(1, int(self.cfg.block_size))
            for i0 in range(0, S, bs):
                i1 = min(S, i0 + bs)
                f_i = f[i0:i1]  # [Bi, N]

                # Since f is L2-normalized, use dot-product for efficient L2 distances.
                dot = f_i @ f.T  # [Bi, S]
                dist2 = (2.0 - 2.0 * dot).clamp_min(0.0)
                dist = torch.sqrt(dist2 + 1e-12)

                ids_i = ids[i0:i1]  # [Bi]
                same = ids_i[:, None] == ids[None, :]  # [Bi, S]
                diff = ~same
                self_mask = (torch.arange(i0, i1, device=device)[:, None] == point_ids[None, :])

                pull = pull + dist[same & (~self_mask)].sum()
                push = push + F.relu(float(self.cfg.margin) - dist)[diff].sum()

            if self.cfg.normalize_by_pairs:
                # Ordered pairs excluding diagonal: S*(S-1)
                denom = float(S * (S - 1))
                pull = pull / denom
                push = push / denom

            total_pull = total_pull + pull
            total_push = total_push + push
            total_pairs = total_pairs + float(S * (S - 1))

        loss = float(self.cfg.lambda_pull) * total_pull + float(self.cfg.lambda_push) * total_push
        loss = float(self.cfg.weight) * loss
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


"""Discriminative instance embedding loss (IGGT-style).

Implements the discriminative loss from De Brabandere et al. (2017):
  * Pull loss: penalise instance pixels whose embeddings deviate from the
    instance mean beyond ``delta_v``.
  * Push loss: penalise pairs of instance means that are closer than
    ``2 * delta_d``.

Unlike the MVC contrastive loss, this operates per-image (not cross-view).

Expected inputs (provided via ``depth_dict`` by the training loop):
  - depth_dict['instance_feat_map']: Float[Tensor] shaped [B, V, N, H, W]
  - depth_dict['instance_mask']:     Int64[Tensor] shaped [B, V, H, W]
  - (optional) depth_dict['instance_valid_mask']: Bool[Tensor] shaped [B, V, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossDiscCfg:
    weight: float = 1.0
    delta_v: float = 0.5
    delta_d: float = 1.5
    ignore_id: int = 0


@dataclass
class LossDiscCfgWrapper:
    disc: LossDiscCfg


class LossDisc(Loss[LossDiscCfg, LossDiscCfgWrapper]):
    """Discriminative instance segmentation loss (IGGT-style)."""

    def __init__(self, cfg: LossDiscCfgWrapper) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Core discriminative loss (operates on a single [C, H, W] feature map
    # with a corresponding [H, W] instance label map).
    # ------------------------------------------------------------------

    @staticmethod
    def _discriminative_loss(
        embedding: Tensor,
        seg_gt: Tensor,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        ignore_id: int = 0,
    ) -> Tensor:
        """Compute discriminative loss for a single image.

        Parameters
        ----------
        embedding : Tensor [C, H, W]
        seg_gt : Tensor [H, W]  (integer instance ids)
        delta_v : float
            Pull hinge margin (per-instance variance).
        delta_d : float
            Push hinge margin (inter-instance distance).
        ignore_id : int
            Instance id to ignore (e.g. background=0).

        Returns
        -------
        Tensor (scalar)
        """
        unique_ids = torch.unique(seg_gt)
        unique_ids = unique_ids[unique_ids != ignore_id]
        num_instances = len(unique_ids)

        if num_instances == 0:
            return torch.tensor(0.0, device=embedding.device, dtype=embedding.dtype)

        mu_list: list[Tensor] = []
        l_var = embedding.new_tensor(0.0)

        for inst_id in unique_ids:
            mask = seg_gt == inst_id
            masked_emb = embedding[:, mask]  # [C, N_pixels]
            mean = masked_emb.mean(dim=1)    # [C]
            mu_list.append(mean)

            dist = torch.norm(masked_emb - mean.unsqueeze(1), dim=0)  # [N_pixels]
            l_var = l_var + torch.clamp(dist - delta_v, min=0).pow(2).mean()

        l_var = l_var / num_instances

        l_dist = embedding.new_tensor(0.0)
        if num_instances > 1:
            mu = torch.stack(mu_list)            # [K, C]
            mu_a = mu.unsqueeze(0)               # [1, K, C]
            mu_b = mu.unsqueeze(1)               # [K, 1, C]
            dist = torch.norm(mu_a - mu_b, dim=2)  # [K, K]

            diag_mask = torch.eye(num_instances, device=embedding.device, dtype=torch.bool)
            off_diag = dist[~diag_mask]

            l_dist = torch.clamp(2 * delta_d - off_diag, min=0).pow(2).mean()

        return l_var + l_dist

    # ------------------------------------------------------------------
    # AnySplat Loss interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.extra_logs: dict[str, Tensor] = {}

        if depth_dict is None:
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        feat_map: Tensor | None = depth_dict.get("instance_feat_map")
        inst_mask: Tensor | None = depth_dict.get("instance_mask")
        valid_mask: Tensor | None = depth_dict.get("instance_valid_mask")

        if feat_map is None or inst_mask is None:
            return torch.tensor(0.0, device=prediction.color.device, dtype=torch.float32)

        B, V, C, H, W = feat_map.shape
        device = feat_map.device

        # Apply valid mask to instance labels (set invalid pixels to ignore_id).
        if valid_mask is not None:
            if valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            inst_mask = inst_mask * valid_mask.long()

        # Flatten batch and view dimensions.
        feat_flat = feat_map.view(B * V, C, H, W)
        mask_flat = inst_mask.view(B * V, H, W)

        total_loss = feat_map.new_tensor(0.0)
        num_valid = 0
        total_var = feat_map.new_tensor(0.0)
        total_dist = feat_map.new_tensor(0.0)

        for i in range(B * V):
            loss_i = self._discriminative_loss(
                feat_flat[i],
                mask_flat[i],
                delta_v=float(self.cfg.delta_v),
                delta_d=float(self.cfg.delta_d),
                ignore_id=int(self.cfg.ignore_id),
            )
            if loss_i > 0:
                total_loss = total_loss + loss_i
                num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid

        loss = float(self.cfg.weight) * total_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.extra_logs = {
            "disc_loss_raw": total_loss.detach(),
        }

        return loss

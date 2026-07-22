"""QueryPhysGMReadout — PhysGM-style property readout on SegVGGT object queries.

The IGGT physics path has to *construct* a per-object token: it masked-average-pools
backbone patch tokens using the GT instance mask (:class:`PhysGMReadout`).  That makes
the readout depend on ground-truth segmentation, so it cannot run at inference on a
new scene.

SegVGGT already carries one vector per object — the object query.  This head therefore
decodes properties straight off ``SegVGGTPrediction.query_embed``:

  * no instance mask, no pooling, no clustering;
  * usable at inference with no GT whatsoever — every query that clears the objectness
    threshold comes with its own (mu, var);
  * the query<->GT correspondence needed for *training* comes from the Hungarian match
    that :class:`~src.loss.loss_segvggt.LossSegVGGT` already computes for the masks.

The decoder itself is byte-for-byte PhysGM's: ``LayerNorm → Linear → GELU → Linear(·,2)``
with a small-init last layer, variance through ``softplus + 1e-2``.  It is imported from
:mod:`.physgm_readout` rather than re-typed so the two schemes cannot drift apart.

Lives on the encoder (learnable). Loss must not own this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.dataset.physics.types import PROPERTY_NAMES

from .physgm_readout import _make_property_decoder


class QueryPhysGMReadout(nn.Module):
    """Per-object-query property readout: ``[B, Q, D] -> (mu, var)`` each ``[B, Q, P]``."""

    def __init__(
        self,
        token_dim: int,
        hidden: int = 64,
        property_names: tuple[str, ...] = PROPERTY_NAMES,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.property_names = property_names
        self.decoders = nn.ModuleList(
            _make_property_decoder(token_dim, hidden) for _ in property_names
        )

    def forward(self, query_embed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        query_embed : [B, Q, D] object-query vectors (D must equal ``token_dim``).

        Returns
        -------
        mu, var : both [B, Q, P] in normalized model space (see PHYSGM_NORMALIZATION
            in src/dataset/physics/parsers.py). ``var`` is a *learned predictive*
            variance, not a regression of GT variance.
        """
        if query_embed.shape[-1] != self.token_dim:
            raise ValueError(
                f"query_embed has dim {query_embed.shape[-1]}, expected {self.token_dim}. "
                "Set encoder cfg `embed_dim` and the readout's token_dim consistently."
            )
        # Force fp32 (repo convention: heads/loss run outside the bf16 autocast).
        with torch.amp.autocast("cuda", enabled=False):
            x = query_embed.float()
            outs = [decoder(x) for decoder in self.decoders]   # P x [B, Q, 2]
            out = torch.stack(outs, dim=2)                     # [B, Q, P, 2]
        # PhysGM variance activation: softplus + floor.
        return out[..., 0], F.softplus(out[..., 1]) + 1e-2

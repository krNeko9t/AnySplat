from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss

logger = logging.getLogger(__name__)


@dataclass
class MvcUniformSamplingCfg:
    # Number of uniformly sampled eligible pixels across all views.
    num_points: int = 0


@dataclass
class MvcInstanceSamplingCfg:
    # Sample K instances uniformly, then M pixels per instance.
    num_instances: int = 0
    samples_per_instance: int = 0
    # Ensure each sampled instance has at least this many pixels (sampling with replacement if needed).
    min_samples_per_instance: int = 2
    # Cap number of within-instance pull pairs per instance (0 => all pairs).
    max_pull_pairs_per_instance: int = 0


@dataclass
class MvcNegativeSamplingCfg:
    # For each anchor, sample these many random negatives (different instance id).
    rand_per_anchor: int = 32
    # For each anchor, sample these many hard negatives from local 2D neighborhood (different instance id).
    hard_per_anchor: int = 32
    # Local neighborhood radius (in pixels) for hard negatives (same view).
    hard_radius: int = 2
    # Number of candidate neighbor attempts per hard negative.
    hard_tries: int = 8


@dataclass
class MvcRecipeCfg:
    uniform: MvcUniformSamplingCfg = field(default_factory=MvcUniformSamplingCfg)
    instance: MvcInstanceSamplingCfg = field(default_factory=MvcInstanceSamplingCfg)
    negatives: MvcNegativeSamplingCfg = field(default_factory=MvcNegativeSamplingCfg)


@dataclass
class LossMvcCfg:
    weight: float = 1.0
    # Legacy: number of uniformly sampled points, then compute ALL pairs (O(S^2)).
    # Prefer `recipe` for scalable, long-tail-friendly sampling.
    num_samples: int = 8192
    margin: float = 1.0
    lambda_pull: float = 2.0
    lambda_push: float = 1.0
    ignore_id: int = 0
    block_size: int = 512
    # If enabled, divide by number of evaluated pairs (unordered, i<j).
    normalize_by_pairs: bool = False
    # Recipe-based sampling (recommended). If set and any of its budgets are > 0, it is used.
    recipe: Optional[MvcRecipeCfg] = None
    # Debug print every N steps (0 disables).
    debug_every_steps: int = 0


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
        # Optional side-channel metrics consumed by ModelWrapper.
        # Always reset to avoid stale values when MVC is skipped.
        self.extra_logs: dict[str, Tensor] = {}

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

        def _use_recipe() -> bool:
            r = self.cfg.recipe
            if r is None:
                return False
            if (r.uniform.num_points > 0) or (r.instance.num_instances > 0 and r.instance.samples_per_instance > 0):
                return True
            return False

        def _pairwise_dist(a: Tensor, b: Tensor) -> Tensor:
            # a,b: [P,N] normalized. Return L2 dist.
            dot = (a * b).sum(dim=-1)
            dist2 = (2.0 - 2.0 * dot).clamp_min(0.0)
            return torch.sqrt(dist2 + 1e-12)

        def _sample_uniform(idx_all: Tensor, num_points: int) -> Tensor:
            if num_points <= 0:
                return torch.empty((0,), dtype=torch.int64, device=device)
            S = min(int(num_points), int(idx_all.numel()))
            if S <= 0:
                return torch.empty((0,), dtype=torch.int64, device=device)
            perm = torch.randperm(idx_all.numel(), device=device)[:S]
            return idx_all[perm]

        def _group_by_instance(idx_all: Tensor, ids_flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # Returns (ids_sorted, idx_sorted, boundaries) where boundaries are start indices of each group.
            ids = ids_flat[idx_all].to(torch.int64)  # [P]
            sort = torch.argsort(ids)
            ids_sorted = ids[sort]
            idx_sorted = idx_all[sort]
            # group boundaries: start indices of each new id
            if ids_sorted.numel() == 0:
                boundaries = torch.empty((0,), dtype=torch.int64, device=device)
            else:
                change = torch.nonzero(ids_sorted[1:] != ids_sorted[:-1], as_tuple=False).squeeze(1) + 1
                boundaries = torch.cat(
                    [torch.zeros((1,), dtype=torch.int64, device=device), change], dim=0
                )
            return ids_sorted, idx_sorted, boundaries

        def _sample_instance_wise(idx_all: Tensor, ids_flat: Tensor, cfg: MvcInstanceSamplingCfg) -> tuple[Tensor, list[Tensor]]:
            """Return sampled pixel indices and per-instance groups (indices into sampled set)."""
            if cfg.num_instances <= 0 or cfg.samples_per_instance <= 0:
                return torch.empty((0,), dtype=torch.int64, device=device), []
            ids_sorted, idx_sorted, boundaries = _group_by_instance(idx_all, ids_flat)
            if boundaries.numel() == 0:
                return torch.empty((0,), dtype=torch.int64, device=device), []

            # Determine group ranges.
            starts = boundaries
            ends = torch.empty_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = int(idx_sorted.numel())
            num_groups = int(starts.numel())

            K = min(int(cfg.num_instances), num_groups)
            perm = torch.randperm(num_groups, device=device)[:K]
            starts_k = starts[perm]
            ends_k = ends[perm]

            sampled_idx_list: list[Tensor] = []
            for s0, e0 in zip(starts_k.tolist(), ends_k.tolist()):
                pool = idx_sorted[s0:e0]
                n = int(pool.numel())
                m = int(cfg.samples_per_instance)
                if n == 0:
                    continue
                if n >= m:
                    pick = pool[torch.randperm(n, device=device)[:m]]
                else:
                    # sample with replacement to reach m (and enforce min_samples_per_instance)
                    need = max(m, int(cfg.min_samples_per_instance))
                    ridx = torch.randint(0, n, (need,), device=device)
                    pick = pool[ridx]
                sampled_idx_list.append(pick)

            if not sampled_idx_list:
                return torch.empty((0,), dtype=torch.int64, device=device), []

            sampled = torch.cat(sampled_idx_list, dim=0).to(torch.int64)
            # Build groups as indices into sampled tensor.
            groups: list[Tensor] = []
            offset = 0
            for pick in sampled_idx_list:
                n = int(pick.numel())
                groups.append(torch.arange(offset, offset + n, device=device, dtype=torch.int64))
                offset += n
            return sampled, groups

        def _build_pull_pairs(groups: list[Tensor], cfg: MvcInstanceSamplingCfg) -> tuple[Tensor, Tensor]:
            # Return (i_idx, j_idx) indices into sampled points tensor.
            i_list: list[Tensor] = []
            j_list: list[Tensor] = []
            for g in groups:
                m = int(g.numel())
                if m < 2:
                    continue
                # all unordered pairs within g
                ii, jj = torch.triu_indices(m, m, offset=1, device=device)
                pi = g[ii]
                pj = g[jj]
                if cfg.max_pull_pairs_per_instance and pi.numel() > int(cfg.max_pull_pairs_per_instance):
                    keep = int(cfg.max_pull_pairs_per_instance)
                    sel = torch.randperm(pi.numel(), device=device)[:keep]
                    pi = pi[sel]
                    pj = pj[sel]
                i_list.append(pi)
                j_list.append(pj)
            if not i_list:
                return (
                    torch.empty((0,), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.int64, device=device),
                )
            return torch.cat(i_list, dim=0), torch.cat(j_list, dim=0)

        def _sample_random_negatives(
            anchors_idx: Tensor, anchors_ids: Tensor, idx_all: Tensor, ids_flat: Tensor, q: int
        ) -> tuple[Tensor, Tensor]:
            """Return (a_idx, b_idx) into sampled points tensor for push pairs with random negatives."""
            if q <= 0 or anchors_idx.numel() == 0:
                return (
                    torch.empty((0,), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.int64, device=device),
                )
            A = int(anchors_idx.numel())
            # Sample candidate negatives from global eligible pixels, then pick first with different id.
            tries = 8
            cand = torch.randint(0, int(idx_all.numel()), (A, q, tries), device=device)
            neg_pix = idx_all[cand]  # [A,q,T]
            neg_ids = ids_flat[neg_pix].to(torch.int64)
            same = neg_ids == anchors_ids[:, None, None]
            ok = ~same
            has = ok.any(dim=-1)  # [A,q]
            # pick first ok along tries dimension
            first = ok.to(torch.int64).argmax(dim=-1)  # [A,q]
            pick_pix = neg_pix[torch.arange(A, device=device)[:, None], torch.arange(q, device=device)[None, :], first]
            # if none ok, fall back to self pixel (will yield dist=0 and push=max(margin,...) but rare)
            pick_pix = torch.where(has, pick_pix, anchors_idx[:, None].expand(-1, q))
            # Map pixel indices to "sampled points" indices: negatives are pixels, but we want negatives among sampled points.
            # For push, we only need features, so we treat push pairs in pixel-index space later.
            # Here, return pixel indices directly (a_pix, b_pix).
            a_pix = anchors_idx[:, None].expand(-1, q).reshape(-1)
            b_pix = pick_pix.reshape(-1)
            return a_pix, b_pix

        def _sample_hard_negatives_2d(
            anchors_idx: Tensor,
            anchors_ids: Tensor,
            elig_flat: Tensor,
            ids_flat: Tensor,
            q: int,
            radius: int,
            tries: int,
        ) -> tuple[Tensor, Tensor]:
            if q <= 0 or anchors_idx.numel() == 0:
                return (
                    torch.empty((0,), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.int64, device=device),
                )
            A = int(anchors_idx.numel())
            HW = H * W
            v = anchors_idx // HW
            rem = anchors_idx - v * HW
            y = rem // W
            x = rem - y * W

            # Propose neighbor offsets.
            r = int(radius)
            # [A,q,T]
            dx = torch.randint(-r, r + 1, (A, q, tries), device=device)
            dy = torch.randint(-r, r + 1, (A, q, tries), device=device)
            nx = (x[:, None, None] + dx).clamp(0, W - 1)
            ny = (y[:, None, None] + dy).clamp(0, H - 1)
            nv = v[:, None, None]
            nidx = nv * HW + ny * W + nx  # [A,q,T]

            ok = elig_flat[nidx] & (ids_flat[nidx].to(torch.int64) != anchors_ids[:, None, None])
            has = ok.any(dim=-1)  # [A,q]
            first = ok.to(torch.int64).argmax(dim=-1)  # [A,q]
            pick = nidx[torch.arange(A, device=device)[:, None], torch.arange(q, device=device)[None, :], first]
            pick = torch.where(has, pick, anchors_idx[:, None].expand(-1, q))
            a_pix = anchors_idx[:, None].expand(-1, q).reshape(-1)
            b_pix = pick.reshape(-1)
            return a_pix, b_pix

        total_pull = torch.tensor(0.0, device=device)
        total_push = torch.tensor(0.0, device=device)
        total_pairs = torch.tensor(0.0, device=device)

        # Debug accumulators (only printed on rank0).
        dbg_pull_pairs = 0
        dbg_push_pairs = 0
        dbg_instances_lt2 = 0
        dbg_instances = 0

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
            # Gather features: [V,N,H,W] -> [P,N]
            feats_flat = feats_b.permute(0, 2, 3, 1).reshape(-1, N)  # [P, N]

            if _use_recipe():
                r = self.cfg.recipe
                assert r is not None

                # Sample anchors/pull points.
                idx_uniform = _sample_uniform(idx_all, int(r.uniform.num_points))
                idx_inst, groups = _sample_instance_wise(idx_all, ids_flat, r.instance)

                # Combine (deduplicate to avoid self-pairs dominating).
                idx_points = torch.cat([idx_uniform, idx_inst], dim=0) if (idx_uniform.numel() + idx_inst.numel()) > 0 else torch.empty((0,), dtype=torch.int64, device=device)
                if idx_points.numel() < 2:
                    continue
                idx_points = torch.unique(idx_points)

                f_pts = F.normalize(feats_flat[idx_points], p=2, dim=-1, eps=1e-8)  # [S,N]
                ids_pts = ids_flat[idx_points].to(torch.int64)

                # Pull pairs: within-instance, based on instance-wise sampled groups.
                pull_i = torch.empty((0,), dtype=torch.int64, device=device)
                pull_j = torch.empty((0,), dtype=torch.int64, device=device)
                if idx_inst.numel() > 0 and groups:
                    # Map original sampled instance indices into the deduped idx_points tensor.
                    # Build lookup from pixel idx -> position in idx_points.
                    # Note: idx_points is unique; idx_inst may have duplicates.
                    # We'll use a hash via sorting.
                    sp = torch.argsort(idx_points)
                    idx_points_sorted = idx_points[sp]

                    def _map_pix_to_pos(pix: Tensor) -> Tensor:
                        pos = torch.searchsorted(idx_points_sorted, pix)
                        pos = pos.clamp(0, idx_points_sorted.numel() - 1)
                        found = idx_points_sorted[pos] == pix
                        # if not found (due to duplicates), mark as -1
                        pos = torch.where(found, pos, torch.full_like(pos, -1))
                        # map back to original order positions
                        return torch.where(pos >= 0, sp[pos], pos)

                    # Rebuild groups as positions in idx_points.
                    groups_pos: list[Tensor] = []
                    for g in groups:
                        pix = idx_inst[g]
                        pos = _map_pix_to_pos(pix)
                        pos = pos[pos >= 0]
                        # Remove duplicates introduced by sampling-with-replacement.
                        pos = torch.unique(pos)
                        if pos.numel() >= 2:
                            groups_pos.append(pos)
                    pull_i, pull_j = _build_pull_pairs(groups_pos, r.instance)

                    dbg_instances = dbg_instances + len(groups_pos)
                    dbg_instances_lt2 = dbg_instances_lt2 + sum(int(g.numel() < 2) for g in groups_pos)

                # Push negatives: anchors are all sampled points.
                a_h, b_h = _sample_hard_negatives_2d(
                    idx_points,
                    ids_pts,
                    elig_flat,
                    ids_flat,
                    q=int(r.negatives.hard_per_anchor),
                    radius=int(r.negatives.hard_radius),
                    tries=int(r.negatives.hard_tries),
                )
                a_r, b_r = _sample_random_negatives(
                    idx_points,
                    ids_pts,
                    idx_all,
                    ids_flat,
                    q=int(r.negatives.rand_per_anchor),
                )
                a_pix = torch.cat([a_h, a_r], dim=0)
                b_pix = torch.cat([b_h, b_r], dim=0)

                # Compute pull/push losses (mean over pairs).
                pull = torch.tensor(0.0, device=device)
                push = torch.tensor(0.0, device=device)
                if pull_i.numel() > 0:
                    dist = _pairwise_dist(f_pts[pull_i], f_pts[pull_j])
                    pull = dist.mean()
                    dbg_pull_pairs += int(dist.numel())

                if a_pix.numel() > 0:
                    f_a = F.normalize(feats_flat[a_pix], p=2, dim=-1, eps=1e-8)
                    f_b = F.normalize(feats_flat[b_pix], p=2, dim=-1, eps=1e-8)
                    dist = _pairwise_dist(f_a, f_b)
                    push = F.relu(float(self.cfg.margin) - dist).mean()
                    dbg_push_pairs += int(dist.numel())

                total_pull = total_pull + pull
                total_push = total_push + push
            else:
                # Legacy uniform sampling + all pairs (unordered i<j), computed in blocks.
                S = min(int(self.cfg.num_samples), int(idx_all.numel()))
                if S < 2:
                    continue

                perm = torch.randperm(idx_all.numel(), device=device)[:S]
                idx = idx_all[perm]  # [S]
                f = feats_flat[idx]  # [S, N]

                # L2 normalize before distance.
                f = F.normalize(f, p=2, dim=-1, eps=1e-8)
                ids = ids_flat[idx].to(torch.int64)  # [S]

                pull = torch.tensor(0.0, device=device)
                push = torch.tensor(0.0, device=device)

                bs = max(1, int(self.cfg.block_size))
                for i0 in range(0, S, bs):
                    i1 = min(S, i0 + bs)
                    f_i = f[i0:i1]  # [Bi, N]

                    dot = f_i @ f.T  # [Bi, S]
                    dist2 = (2.0 - 2.0 * dot).clamp_min(0.0)
                    dist = torch.sqrt(dist2 + 1e-12)

                    ids_i = ids[i0:i1]  # [Bi]
                    same = ids_i[:, None] == ids[None, :]  # [Bi, S]
                    diff = ~same
                    i_idx = torch.arange(i0, i1, device=device)[:, None]  # [Bi,1]
                    j_idx = torch.arange(S, device=device)[None, :]       # [1,S]
                    upper = j_idx > i_idx                                  # [Bi,S]

                    pull = pull + dist[same & upper].sum()
                    push = push + F.relu(float(self.cfg.margin) - dist)[diff & upper].sum()

                if self.cfg.normalize_by_pairs:
                    denom = float(S * (S - 1) / 2.0)
                    pull = pull / denom
                    push = push / denom

                total_pull = total_pull + pull
                total_push = total_push + push
                total_pairs = total_pairs + float(S * (S - 1) / 2.0)

        # Decompose into pull/push components for easier debugging.
        pull_term = float(self.cfg.weight) * float(self.cfg.lambda_pull) * total_pull
        push_term = float(self.cfg.weight) * float(self.cfg.lambda_push) * total_push
        loss = pull_term + push_term

        pull_term = torch.nan_to_num(pull_term, nan=0.0, posinf=0.0, neginf=0.0)
        push_term = torch.nan_to_num(push_term, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Expose both weighted and raw terms.
        self.extra_logs = {
            "mvc_pull": pull_term.detach(),
            "mvc_push": push_term.detach(),
            "mvc_pull_raw": total_pull.detach(),
            "mvc_push_raw": total_push.detach(),
        }

        if int(self.cfg.debug_every_steps) > 0 and (global_step % int(self.cfg.debug_every_steps) == 0):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            if rank == 0 and _use_recipe():
                logger.info(
                    "[mvc] step=%s pull_pairs=%s push_pairs=%s instances=%s instances_lt2=%s",
                    global_step,
                    dbg_pull_pairs,
                    dbg_push_pairs,
                    dbg_instances,
                    dbg_instances_lt2,
                )

        return loss


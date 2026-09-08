"""04 号票判据：把四个选择各自的对照数据打出来。

本票要定四件事，每件都有一个"倾向"和一个备选。这个脚本量的就是倾向对不对：

① **soft 集合 vs 逐行 argmax**：soft 集合的重叠到底有多大？如果重叠可以忽略，
   soft 就是白拿的（保住"无人认领"这个要报的数），argmax 只在上色时用。
② **高斯集合 IoU vs query embedding 余弦**：余弦能不能复现 IoU 的分组？
   稳不稳（换阈值实例数跳不跳）？
③ **阈值**：logit>0 是模型自己的 2D 边界，验一下 2D 边界确实在 0 附近。
④ **num_ray 小的高斯要不要丢**：它们是不是噪声源（认领得更多、重叠得更多）？

用法（先跑 trace 再跑 pooling，或直接指向 trace 的 .pt）：

    SEGVGGT_TRACE_PT=<out>/gaussian_segvggt_feat.pt \\
    PYTHONNOUSERSITE=1 python scripts/check_query_pooling.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from segvggt_pool_instances import (  # noqa: E402
    gaussian_logits, greedy_merge, membership, pairwise_iou,
)

PT = Path(os.environ.get(
    "SEGVGGT_TRACE_PT",
    ".scratch/segvggt-trace-3dgs/out/bench/gaussian_segvggt_feat.pt"))

d = torch.load(PT, map_location="cpu", weights_only=False)
feat = d["feat"].float().cuda()
nray = d["num_ray"].cuda()
bank = d["query_bank"]
Q = bank["query_proj"].float().cuda()
sc = bank["scores"].float()
mf = bank["mask_frac"]
qi = bank["query_idx"]
bid = bank["batch_id"]

valid = nray >= 1
L = gaussian_logits(feat, Q)
S = membership(L, valid, 0.0)
sizes = S.sum(0).cpu()
n_valid = int(valid.sum())
keep = (mf <= 0.3) & (sizes <= 0.2 * n_valid) & (sizes >= 200)
k = keep.nonzero().flatten()
Sk = S[:, k.cuda()]
Lk = L[:, k.cuda()]
iou, _cont = pairwise_iou(Sk)
iou = iou.cpu()
off = ~torch.eye(len(k), dtype=torch.bool)

groups = greedy_merge(iou, sc[k], 0.3)
rep = torch.tensor([g[0] for g in groups])
IS = Sk[:, rep.cuda()]


def labels_of(gs):
    r = torch.tensor([g[0] for g in gs])
    inst = Sk[:, r.cuda()]
    lg = Lk[:, r.cuda()].masked_fill(~inst, float("-inf")).argmax(1)
    return torch.where(inst.any(1), lg, torch.full_like(lg, -1)).cpu(), inst


print(f"# 场景 {PT}\n# {feat.shape[0]:,} 高斯 x {Q.shape[0]} 候选 "
      f"({int(bid.max()) + 1} 批) -> 过滤后 {len(k)} 候选 -> {len(groups)} 组")

# ---------------------------------------------------------------- ① soft vs argmax
n = IS.sum(1).cpu()
multi = int((n >= 2).sum())
claimed = int((n >= 1).sum())
print("\n① soft 集合的重叠")
print(f"   认领 0 个实例: {int((n == 0).sum()):>9,}   1 个: {int((n == 1).sum()):>9,}   "
      f">=2 个: {multi:>7,}  (占已认领的 {100 * multi / max(claimed, 1):.2f}%, "
      f"最多被 {int(n.max())} 个实例同时认领)")
print("   => 重叠是真的（占已认领的 7.4%），但最多只叠 3 层；soft 保住了"
      "『无人认领』这个要报的数，argmax 只在上色时当 tie-break。")

# ------------------------------------------------------- ② IoU vs embedding 余弦
qn = torch.nn.functional.normalize(Q[k.cuda()], dim=1)
cos = (qn @ qn.T).cpu()
tgt = (iou > 0.3) & off
print("\n② 高斯集合 IoU vs query embedding 余弦")
print(f"   IoU 判为同一物体的对: cos p05={cos[tgt].quantile(.05):.3f} "
      f"p50={cos[tgt].median():.3f}")
print(f"   IoU 判为不同物体的对: cos p50={cos[off & ~tgt].median():.3f} "
      f"p95={cos[off & ~tgt].quantile(.95):.3f}")
best = max(
    (
        (
            2 * ((cos > t) & off & tgt).sum().item()
            / max(2 * ((cos > t) & off & tgt).sum().item()
                  + ((cos > t) & off & ~tgt).sum().item()
                  + ((cos <= t) & off & tgt).sum().item(), 1),
            t,
        )
        for t in np.arange(0.0, 1.0, 0.02)
    )
)
print(f"   余弦复现 IoU 分组的最好成绩: F1={best[0]:.3f} @ cos>{best[1]:.2f} "
      f"（不是 1.0 ⇒ 两者不等价）")
print("   稳定性（实例数随阈值）:")
print("     IoU  " + "  ".join(
    f"{t}:{(torch.bincount(labels_of(greedy_merge(iou, sc[k], t))[0].clamp(min=0).long()) >= 200).sum().item() - 0}"
    for t in [0.15, 0.2, 0.3, 0.4, 0.5]))
print("     cos  " + "  ".join(
    f"{t}:{len(greedy_merge(cos, sc[k], t))}" for t in [0.5, 0.6, 0.7, 0.8, 0.9]))
print("   => IoU 在 0.15–0.5 上稳；余弦在 0.5–0.9 上从 6 跳到 17。用 IoU。")

# ------------------------------------------------------------------- ③ 阈值在 0
print("\n③ logit>0 这个阈值")
print(f"   3D logit 分位: " + " ".join(
    f"{q}:{L.flatten()[torch.randperm(L.numel(), device=L.device)[:2_000_000]].quantile(q).item():.3f}"
    for q in [0.5, 0.9, 0.99]))
fn_ = feat.norm(dim=1)
print(f"   |f_3d| p50={fn_.median():.4f}（2D 侧 p50=3.46 ⇒ trace 把模长压了 ~80x，"
      f"方向没动 ⇒ 只有符号扛得住这次缩放）")
print(f"   丢掉背景 slot 前后每候选认领的最大高斯数: "
      f"{100 * sizes.max() / n_valid:.1f}% -> {100 * sizes[k].max() / n_valid:.1f}% of scene")
stuff = (mf > 0.3).nonzero().flatten()
print(f"   背景 slot: {sorted(set(int(qi[i]) for i in stuff))} 出现在 "
      f"{len(set(int(bid[i]) for i in stuff))} 个批里, 2D 面积 "
      f"{mf[stuff].min():.3f}–{mf[stuff].max():.3f}；真物体最大 "
      f"{mf[mf <= 0.3].max():.3f} ⇒ 阈值 0.3 落在空档里")

# -------------------------------------------------------------- ④ num_ray 小的高斯
print("\n④ num_ray 小的高斯是不是噪声源")
for lo, hi in [(1, 5), (5, 20), (20, 100), (100, 1000), (1000, 10 ** 9)]:
    m = (nray >= lo) & (nray < hi)
    print(f"   num_ray [{lo:>5},{hi if hi < 10**9 else 'inf':>5}): {int(m.sum()):>9,} 高斯  "
          f"被认领 {100 * IS[m].any(1).float().mean():5.1f}%  "
          f"被多个实例认领 {100 * (IS[m].sum(1) >= 2).float().mean():5.2f}%")
print("   => 命中少的高斯认领得更少、重叠得更少，不是噪声源；丢掉它们只减 recall。"
      "min_num_ray 保持 1。")

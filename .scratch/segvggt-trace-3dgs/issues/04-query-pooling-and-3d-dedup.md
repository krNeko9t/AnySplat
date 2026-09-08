# 04 — 跨批 query 池化 + 3D IoU 去重

Type: task
Status: open
Blocked by: [01 — SegVGGT 的 feature map 跨 forward 漂不漂](01-cross-batch-feature-consistency.md), [03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)
Blocks: [06 — 物性数值怎么摆才不撒谎](06-how-to-present-physics-honestly.md), [07 — 3dovs 上顺路报分割指标](07-3dovs-segmentation-metric.md), [08 — 在真正的 3DGS 场景上验后端](08-3dgs-backend-on-garden.md), [09 — 验收交付什么](09-acceptance-artifacts.md)
Assignee: —

> 这是把"一堆碎片"变成"一堆实例"的那一步，也是本图 Destination 的主体。

## Question

03 号票跑完，手上是：全局的 `gau_feat [P,128]`，和 M = Σ_b N_b 个候选 query
（每行自带 128 维投影、score、和它那一行物性）。M 在 25 批 × ~15 的量级 ≈ 几百。

单批 query 不覆盖场景（一批 4 视角只含那 4 个视角里的物体，地图核心论证），
所以要把所有批并起来再去重：

```
Q_all  = concat_b(Q_b)                    # [M, 128]
logits = gau_feat @ Q_all.T               # [P, M]
```

`logits` 每一列是一个候选实例在**全部** P 个高斯上的响应——包括它自己那批视角
从没见过的高斯。然后在 3D 里合并。

## 要定的事（本票要答的，不是已知的）

1. **高斯归属怎么定**：逐列 `sigmoid > thr` 得到软集合，还是逐行 `argmax` 得到硬划分？
   前者允许重叠和"无人认领"，后者保证划分完备但强行给每个高斯派一个主。
   建议先做前者，"无人认领"的比例本身就是要报的数（→ 地图 Not yet specified）。
2. **合并判据**：候选之间按**高斯集合 IoU** 合并（阈值待定）。
   备选是按 `Q_all` 的 embedding 余弦合并 —— 但那等于又信了一遍跨批一致性，
   而 IoU 是在 3D 里直接看重叠，独立于 01 号票的结论。**倾向 IoU。**
3. **合并组的物性怎么算**：按 score 加权平均 mu？还是取 score 最高那一行？
   `var` 怎么合？（注意 `var` 是学出来的预测方差，不是 GT 方差的回归 ——
   见 `query_physgm_readout.py` 的 docstring，别当成置信区间乱用。）
4. **归一化**：`gau_feat = sum_gau_sem / num_gsem.clamp(min=1)`。
   `num_gsem` 很小的高斯（只被一两条光线打到）要不要直接丢？

## 判据

`3dovs/bench` 上出一张 3D 分割上色图 + 一张实例列表（实例 id / 高斯数 / score / E,ν,ρ）。
**先看图像不像话，再谈指标**（指标归 07 号票）。

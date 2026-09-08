# 07 — 3dovs 上顺路报 class-agnostic 3D 分割指标

Type: task
Status: open
Blocked by: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Blocks: —
Assignee: —

> **顺路的，不是终点。** 地图 Destination 明写判据是"链路跑通且出得了图"。
> 这票只在 04 出图之后花很少的力气去捡一个量化数；捡不到不影响本图到达终点。

## Question

`3dovs/{bench,bed,room}` 自带 `id_maps`（多视角一致的整数 ID 图），
`configs/trace/bench_gt_idmap.json` 这条路以前跑通过 —— 也就是说
**把 GT ID 图 trace 到高斯上得到 GT 的 3D 实例划分**，这套装置现成。

于是能对：04 号票产出的高斯划分 vs GT trace 出来的高斯划分，
报 class-agnostic 的 3D 实例指标（mAP / AP50 / AP25，或最朴素的 mean IoU + 匹配率）。

## 要小心的

- **两边都是 trace 出来的**，共享同一套相机和同一份几何 ⇒ 这个对比测的是
  "特征/query 好不好"，**不测 trace 本身**。这是优点（变量干净），但结论的措辞要跟上：
  不能说成"3D 实例分割 SOTA"，只能说"在同一 trace 装置下，本方法 vs GT 的一致性"。
- GT `id_maps` 自己的覆盖率如何？没被任何 id 覆盖的高斯要不要计入分母。
- `bed` / `room` 也在本机，多跑两个场景几乎零成本，但三个场景不构成 benchmark，
  报的时候别写成表格假装是 benchmark。

## 判据

一个数（或三个场景三个数）+ 一句能写进论文且不夸大的措辞。


---

## ⚠️ 05 号票（2026-09-08）纠正的 GT 口径

**`id_maps/*.npy` 不是 GT。** 它是 **336×504**，长宽比 1.5，而原图 1008×756 是 1.333 ——
对不上，因为它是**上一次 IGGT 跑出来的产物**（`iggt_idmap.py:559` 写的就是 `id_maps/`），
分辨率是 IGGT 自己 504×336 预处理的产物。

**逐视角 GT 是 `sam/mask/*.png`**：原生 **756×1008**，8 个实例，id 0 为背景。
本票的指标必须在这一份上算，且**在原生分辨率上算**（预测 mask 上采样过去，不是把 GT 降下来）。

05 号票用它读到的 2D 参考读数（view 00、4 视角一批、`score_thr=0.25`、`mask_thr=0.4`）：
**mean best IoU ≈ 0.613**（252×448 运行点）。本票的 3D 指标应当能和这个数对得上量级。

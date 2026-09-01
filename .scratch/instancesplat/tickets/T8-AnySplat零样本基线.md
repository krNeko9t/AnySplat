---
id: T8
title: AnySplat 零样本基线
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

**HITL：需要人在集群上跑。**

测出 `hf:lhjiang/anysplat` 预训练权重在我们的 val 集（`processed_scannetpp_v2` +
`processed_re10k` 的 held-out 部分）上的**零样本重建指标**：PSNR / SSIM / LPIPS。

为什么必须有这个数：地图的验收门是**绝对**的（disc 路线已注销，没有相对基线）。
「重建不退化」这一条如果没有起点数字，训完根本无从判断——InstanceSplat 加了三个新 loss
并解冻了 backbone，重建质量掉下去是完全可能的，论文 Table 1 里 AnySplat 的 PSNR
（20.76 / 20.73 / 21.14 / 21.53）就是拿来做这个对照的。

要做的：

1. 确定 val 划分（哪些场景 held-out），**写死并记录**，之后所有评测都用同一批
2. 用现有 `AnySplatWrapper` 的 validation 路径跑一遍（`instance_feat_dim: 0`，纯重建）
3. 按视角数 2 / 4 / 8 分别报，和论文 Table 1 的行对齐着看

**产出**：一张表（视角数 × PSNR/SSIM/LPIPS）+ val 场景列表的落盘路径。
这张表会成为 T10 判读结果时的「不退化」门槛。

**注意**：这张票和 T1–T7 完全无关，任何时候都能跑，**建议第一个动手**——它是唯一一张
不依赖任何代码改动、又必须在集群上排队的票。

## R3 关闭后追加：顺手带两个探针（零代码）

这张票是**唯一一张不依赖任何代码改动、又必须在集群上排队**的票，人已经在 GPU 前面了，
顺手把 R3 findings 里两个「只能实测」的数字取回来，能消掉后面一大片不确定性：

1. **`voxelize_ratio`** —— R3 的显存估算里**不确定度最大的一项**（区间 `[0.16, 0.6]`，
   压缩比主要由视角间重叠决定）。仓库**已经在打这个点**：`anysplat.py:645` →
   `anysplat_wrapper.py:142`，**零代码，日志里读数就行**。
   按视角数 2 / 4 / 8 各记一次（正好和本票的三档对齐）。

2. **峰值显存**（`torch.cuda.max_memory_allocated`）按同样三档记一次。
   这是零样本、`instance_feat_dim: 0`、且 backbone 冻结的下界，不等于训练峰值——
   但它能把 R3 那张分项表里的「静态 + 激活」基数**钉死一个真实锚点**，
   让 T12 做完之后的估算不再全靠推算。

这两个数会直接回填 [R3](R3-40G显存预算.md) 的估算区间，并决定
[T12](T12-DPT栈梯度检查点与前向瘦身.md) 做完后是否还需要降视角数。
**它们不改变本票的验收产出**（PSNR/SSIM/LPIPS 表），只是搭便车。

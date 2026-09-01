---
id: R1
title: 实例 margin 的取值
type: wayfinder:research
status: closed
assignee: krNeko9t (research agent)
blocked-by: []
---

## Question

论文给了 $\lambda_{pull},\lambda_{push},\lambda_{cross} = (2,1,2)$ 和 $\lambda_{ins}=0.01$，
但**三个 margin $\delta_{pull}$ / $\delta_{push}$ / $\delta_{cross}$ 一个数值都没给**
（Eq.5 / Eq.6 / Eq.7 只说是 "margin hyperparameters"）。这是整个复现里最大的数值空白，
而 margin 直接决定 pull/push 的死区——取错了 loss 会在训练早期就饱和到 0 或者永远压不下去。

要查清楚并给出取值建议：

1. **De Brabandere et al. 2017**（判别式实例嵌入的原始出处，论文 Eq.5/6 的形式来自它）
   的 $\delta_v$ / $\delta_d$ 常用值是多少？注意本仓库 `config/loss/disc.yaml` 已经用了
   `delta_v: 0.5` / `delta_d: 1.5`，那是**未归一化**嵌入上的取值。
2. **关键差异**：InstanceSplat 的特征是 **$\ell_2$ 归一化后**的（Eq.4 的 `norm(·)`），
   所以所有距离都落在 $[0, 2]$ 区间内，原始论文那套 (0.5, 1.5) 的尺度**不能直接搬**。
   在单位球面上，$\delta_{push}$ 取多大才是「充分分离」？（提示：两个正交的单位向量距离
   $\sqrt{2}\approx1.41$，反向的距离 2。）
3. 查同样在归一化嵌入上做原型对比的近邻工作（Gaussian Grouping、Contrastive Lift、
   OpenGaussian、IGGT 的民间复现）实际用的 margin。
4. $\delta_{cross}$（跨视角原型对齐的容忍度）应该比 $\delta_{pull}$ 大还是小？给出理由。

**产出**：三个数值 + 每个数值的依据 + 一句「如果训练早期 loss 立刻塌到 0 该往哪调」的说明。

## 解决

**三个数值**：`δ_pull = 0.2`、`δ_push = 1.0`、`δ_cross = 0.3`（量纲 = ℓ2 归一化后的欧氏弦长，值域 [0,2]；
论文 Eq.5/6/7 三处都写 `‖·‖₂`，余弦只用在边界分支 Eq.9）。

- **δ_push = 1.0** — IGGT（arXiv:2510.22706 附录 A.3）在**同样是 ℓ2 归一化的 8 维实例特征**上用 `M = 1.0`，
  且 λ_pull/λ_push = 2.0/1.0 与 InstanceSplat 的 (2,1) 逐位相同；几何上 1.0 = 60° 分离，
  8 维 kissing number 240 → 一个视角放得下 240 个实例，而 √2 只能放 16 个（两两内积非正上限 2n）。
  本仓库 `src/loss/loss_mvc.py` + `config/loss/mvc.yaml:4` 已经就是这个值，**不是新引入的超参**。
- **δ_pull = 0.2** — De Brabandere 的强条件「δ_d > 2δ_v」换算到 Eq.6 的量纲（Eq.6 的 margin **不乘 2**，
  与原论文的 `2δ_d` 差一倍）后是 `δ_pull < δ_push/4 = 0.25`；De Brabandere 自身比例 r/S = 1/6 → 0.167。
  取 0.2，在天花板下留余量。**注意 IGGT 的 pull 项无死区（等价 0），所以偏小是安全方向。**
- **δ_cross = 0.3** — 必须 **>** δ_pull：跨视角原型多扛三种 δ_pull 不承担的噪声（可见集不相交、视角相关高光、
  alpha 合成权重差异），用更严的尺去量更吵的量，会让 λ_cross=2 的持续梯度打进共享 3D 高斯、破坏稳定性；
  必须 **<** 2·δ_pull：否则两原型落在实例自身散布的对立两端仍零损失，L_cross 退化成恒零项。
  区间 (0.2, 0.4) 取中点。交叉校验：合并簇有效半径 ≈ 0.25，间距/簇径 = 2:1，正是 HDBSCAN 舒服的比例。

**早期 loss 立刻塌到 0 怎么调**：先分清是哪一项 —— **只有 push 塌 = δ_push 太小，往上调 1.0→1.2（硬上限 1.41，
再高就撞 16 实例的几何天花板）；只有 pull/cross 塌 = 死区太宽，往下调 δ_pull→0.1、δ_cross→0.15
（不要靠调大 δ_push 补救）；三项同时塌 = 不是 margin 问题，去查 mask 是否全空、`ignore_id=0` 是否吞掉全部像素、
`|K_i| ≥ 2` 是否成立。** 注意健康的 step 0 应该三项都显著非零（Eq.4 的原型**不再归一化**，初始时模长 ≈ 1/√|Ω| ≈ 0，
所有原型挤在原点 → push 满值、pull ≈ 1−δ_pull），所以任何一项 step 0 就是 0 都值得怀疑。
判塌缩必须同时看 `mean_k ‖f̄_k‖` 与原型两两距离均值（配合 T5 看板），只看 loss 分不清「学好了」和「塌缩了」。

**置信度**：δ_push 高（一次来源 + 独立几何论证双重收敛）；δ_pull 中、δ_cross 中低 —— **这两个没有任何论文或代码
给过数值**，是约束区间内的自洽取点，建议区间 δ_pull ∈ [0.1, 0.25]、δ_cross ∈ [0.2, 0.4]，调参时**在区间内调**。
三个 δ 论文都没给值，谈不上偏离；未引入论文之外的任何新超参。

**findings**：[`../research/R1-margin-取值.md`](../research/R1-margin-取值.md)

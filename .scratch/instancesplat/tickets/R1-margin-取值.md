---
id: R1
title: 实例 margin 的取值
type: wayfinder:research
status: open
assignee: -
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

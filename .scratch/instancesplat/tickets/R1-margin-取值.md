---
id: R1
title: 实例 margin 的取值
type: wayfinder:research
status: closed
assignee: codebuddy-main
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

**结论：$\delta_{pull}=0.1$，$\delta_{push}=1.0$，$\delta_{cross}=0.05$（ℓ2 归一化嵌入、论文 Eq.5–7 线性 hinge 口径）。**

依据：

1. **尺度换算**：De Brabandere 2017 的 $\delta_v=0.5$ / $\delta_d=1.5$（push 实际 margin = $2\delta_d=3.0$）
   是**未归一化**嵌入上的取值（本仓 `config/loss/disc.yaml:3-4` 与 `src/loss/loss_disc.py:46-49`
   同款，且 `loss_disc.py:109` 的归一化被注释、用原始欧氏距离）。归一化后距离落在 $[0,2]$，
   $d^2=2(1-\cos\theta)$，那套数值不可直接搬。
2. **$\delta_{push}=1.0$**：对应 $\cos\le 0.5$（原型至少相距 60°）。几何上界：正交 $=\sqrt2\approx1.41$；
   margin 超过 $\sqrt2$ 就是要求「比正交更远」，一个场景几十个实例在 8 维球面上无法同时满足
   → push 死区永远压不下去。三个独立来源收敛到 1.0：
   - IGGT（InstanceSplat 同组、数据同源）论文原文即 $\lambda_{pull}:\lambda_{push}=2:1$、**M = 1.0**
     （GitHub `lifuguan/IGGT_official` issue #24 引述论文）；
   - 本仓 `src/loss/loss_mvc.py:62-64`：ℓ2 归一化嵌入、线性 hinge、`margin: 1.0`、$\lambda=(2,1)$；
   - 归一化嵌入度量学习惯例（球面上「充分分离」≈ 余弦 ≤ 0.5）。
3. **$\delta_{pull}=0.1$**：pull 死区作用于像素→原型距离。单位球面上实例内像素散布应远小于 1；
   本仓 mvc 的 pull 干脆无 margin（$\delta=0$，`loss_mvc.py:412`）。论文显式保留了 $\delta_{pull}$，
   给一个小的非零死区来容忍边界像素的合法偏离：0.1（cos ≥ 0.995 不罚）。取 ≥0.5 会让 pull
   在训练中期就失压（梯度消失）。
4. **$\delta_{cross}=0.05 < \delta_{pull}$**：原型是实例内归一化像素的均值（Eq.4），平均掉了逐像素
   噪声；同一实例跨视角原型的漂移理应**小于**像素对自身原型的散布。若 $\delta_{cross}\ge\delta_{pull}$，
   则 pull 收敛后 cross 项恒在死区内 = 白加一项。取 $\delta_{pull}$ 的一半。
5. **早期塌到 0 的调法**：loss = 0 意味着「所有像素都在 $\delta_{pull}$ 内 且 所有原型对都超过
   $\delta_{push}$」——margin 对当前特征尺度太松。先降 $\delta_{pull}$（0.1→0.05→0），再升
   $\delta_{push}$（1.0→1.2，勿超 $\sqrt2$）。反向症状（push 恒正、压不下去）则把 $\delta_{push}$
   降到 0.8，**不动 λ**（λ 是论文给的）。

**偏离说明**：论文未给数值，以上是由球面几何 + IGGT 同组先例 + 本仓 mvc 先例做出的**有据外推**，
不是论文原值；若 T9 smoke run 暴露量级问题，按第 5 条的方向调。

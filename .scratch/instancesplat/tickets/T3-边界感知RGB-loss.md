---
id: T3
title: 边界感知 RGB loss
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

实现论文 3.3 的 $L_{bd\text{-}rgb}$（Eq.9–11）。**这张票不依赖 T1**——边界分数是从
$S_i$（渲染**前**的 2D pixel-aligned 特征图，即 `EncoderOutput.instance_feat_map`）算的，
原文明写 "before Gaussian rasterization"，那个张量现在就有。

公式（参数论文全给了，一个都别自己定）：

- Eq.9：$z_i(p) = \text{norm}(S_i(p))$，取与**四连通**邻居的最大余弦距离 $r_i(p)$。
- Eq.10：在 $5\times5$ 窗口（$w=2$）内取 $r$ 的最大值，$b_i(p) = \sigma\big((\max r - \tau)/T\big)$，
  $\tau = 0.15$，$T = 0.05$。
- Eq.11：$\frac{1}{N}\sum_i \frac{\sum_p b_i(p)\rho_i(p)}{\sum_p b_i(p)}$，
  $\rho_i$ = 逐通道平均的 Charbonnier 差异，$\epsilon = 10^{-3}$。
- 权重 $\lambda_{bd} = 0.02$。

**必须 detach $b_i$**。论文 3.3 末尾 "Instance and semantic source features are detached in
their respective coupling paths" 管的是整节 coupling path。不 detach 的话，模型可以靠把
$S_i$ 抹平来把权重从高误差像素上挪走 —— 纯作弊通道，而且是那种「loss 在降但东西在变差」
的隐蔽失败。

**冷启动行为**（已分析清楚，写进测试当断言）：随机初始化时 8 维特征的邻域余弦距离普遍
接近 1，$b_i = \sigma((1-0.15)/0.05) \approx 1$ 处处饱和；Eq.11 的分母 $\sum_p b_i$ 会把它
归一化掉，整项**退化成均匀加权的 Charbonnier RGB loss**。这是预期行为，不是 bug，
不要加任何 warmup 或延迟启用。

**验证**（CPU 合成张量）：特征处处相同 → $b$ 处处最小；人造一条特征突变边 → $b$ 在边上峰值、
边内低（对应论文 Figure 3）；$b$ 无梯度（detach 生效）；随机初始化下退化成均匀 Charbonnier
（与直接算的均匀 Charbonnier 数值对拍）。

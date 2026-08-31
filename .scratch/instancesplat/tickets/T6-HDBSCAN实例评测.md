---
id: T6
title: HDBSCAN 实例评测
type: wayfinder:task
status: open
assignee: -
blocked-by: [T1]
---

## Question

实现论文推理侧的实例解码（3.3 末段），作为**里程碑门**——T5 的三个标量回答「训练在不在动」，
这张票回答「学出来的东西到底能不能用」。

- 渲染实例特征 → 逐像素 $\ell_2$ 归一化 → **HDBSCAN** 聚类（论文用的就是它，
  McInnes et al. 2017）→ 得到预测实例区域 $\hat{\Omega}_k^{(i)}$。
- 与 GT instance mask 做最优 1-1 匹配，报 **mIoU** 与 **Acc@0.25**（论文 Table 2 的口径：
  mean mask IoU + IoU 超过 0.25 的实例占比）。

要决策的：

- **HDBSCAN 的超参本身会污染指标**（`min_cluster_size` / `min_samples` / `cluster_selection_epsilon`）。
  定一组固定值并锁死，让指标的变化只反映特征质量。给出选取依据（比如 `min_cluster_size`
  按图像像素数的百分比而不是绝对值，才能跨分辨率可比）。
- 在**哪些视角**上评：context view（自重建，与训练同分布）还是 held-out 视角（真泛化）？
  论文 Table 2 是在标注视角上评、拿邻近帧当输入。倾向 held-out，但要有 GT 位姿。
- HDBSCAN 是 CPU 算法且随高斯/像素数增长很慢（论文自己在 Conclusion 里承认这是 limitation）。
  确认它只在 validation/inference 跑，绝不进训练 loop。

**依赖**：`hdbscan` 或 `sklearn.cluster.HDBSCAN`（sklearn ≥1.3 自带）。查 `requirements.txt`
现有版本，**优先用 sklearn 自带的**，少加一个依赖。

**验证**（CPU 合成张量）：人造 3 个分离良好的特征簇 → 聚出 3 类且 mIoU ≈ 1；
全随机特征 → mIoU 接近 0 而不是崩溃；空场景不报错。

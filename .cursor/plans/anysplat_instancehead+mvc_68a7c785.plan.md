---
name: AnySplat InstanceHead+MVC
overview: 在 AnySplat（加载 pretrained weight）基础上，增加 Instance Head 输出 N 维特征，并实现 MVC 对比损失：在每个 batch 内随机采样 N 个像素点（可跨视角），对特征做 L2 归一化后用 L2 距离计算 pull/push（margin=1.0，λ_pull=2.0，λ_push=1.0），从而训练可用于对最终 3DGS 做实例分割的 embedding。
todos:
  - id: data-instance-mask
    content: 在 dataset 与 `BatchedViews` 中接入 `instance_mask` 并保证与 `image` 的 resize/crop 完全对齐
    status: completed
  - id: encoder-instance-head
    content: 在 `EncoderAnySplat` 增加 Instance Head 输出 `instance_feat_map`，并在 voxelize/采样阶段聚合得到 `gaussian_instance_feat`
    status: completed
  - id: mvc-loss
    content: 实现 `LossMVC`：随机采样 N 点、L2 normalize、L2 distance、margin=1.0、λ_pull=2.0、λ_push=1.0，并用 chunk 做全 pairwise 累加
    status: completed
  - id: train-plumbing
    content: 在 `ModelWrapper.training_step` 里把 context+target views 的 mask 与 instance features 组织好并喂给 MVC loss；补充必要日志与 sanity checks
    status: completed
  - id: export-and-infer
    content: 增加 gs-wise embedding 的导出与一个最小聚类/可视化脚本，验证“对 3DGS 直接实例分割”的闭环
    status: completed
isProject: false
---


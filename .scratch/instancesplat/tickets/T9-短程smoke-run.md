---
id: T9
title: 短程 smoke run
type: wayfinder:task
status: open
assignee: -
blocked-by: [T7, T5]
---

## Question

**HITL：需要人在集群上跑。**

投那 15 小时之前，先跑 **500–1000 steps** 确认这套东西是活的。

必须看到的：

1. **不 OOM**，且峰值显存与 R3 的预算一致（不一致说明 R3 算错了，要回头修）
2. **T5 的三个标量在动**：`ins_intra` 下降、`ins_inter_min` 上升、`ins_cross` 下降。
   哪怕幅度很小，方向必须对。
3. `ins_valid_ratio` 稳定在合理水平（不是接近 0）
4. 每步耗时，用来把「10k steps 要跑多久」从估算变成实测
5. RGB loss 没有爆炸——backbone 刚解冻 + 新 loss 接入，是最容易崩的时刻

**判读规则（提前定好，避免事后找理由）**：

- 三个标量方向都对 → 放行 T10
- `ins_cross` **不降**但另两个降 → 强烈指向 R2 的跨视角 id 假设有问题，或 T1 渲染接错了。
  回头查，**不要**靠调 $\lambda_{cross}$ 掩盖
- 三个都不动 → 检查梯度是否真的回到了 `PartHead`（T1/T2 的验证里有这条，但真数据上要复查
  优化器分组是不是把它归进了 ×0.1 组或者压根没进 optimizer）
- OOM → 按 R3 给的砍法顺序砍（视角数上限 → 分辨率 → 梯度检查点），**一次只砍一个**并记录

**产出**：一份 500–1000 step 的曲线截图/tfevents（`scripts/read_tfevents.py` 已有），
外加一句明确的放行/打回结论。

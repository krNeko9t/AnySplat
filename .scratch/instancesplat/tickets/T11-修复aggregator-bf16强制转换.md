---
id: T11
title: 修复 aggregator bf16 强制转换（backbone 实为冻结）
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

**这张票是 R3 挖出来的、比显存更要命的问题，且它直接决定整个 effort 的前提成不成立。**

`src/model/arch/anysplat.py:124`：

```python
self.aggregator = model_full.aggregator.to(torch.bfloat16)
```

这是**永久 dtype 转换**，不是 autocast。后果链：

1. 参数是 bf16 → AdamW 的 exp_avg / exp_avg_sq 也在 bf16，参数更新 `p -= lr * ...` 在 bf16 里做。
2. bf16 尾数 8 位，相对 ULP ≈ 2^-8 ≈ 3.9e-3。典型权重量级 0.02–0.03 → 绝对 ULP ≈ 0.8e-4 ~ 1.2e-4。
3. 地图锁定 transformer lr = base 2e-4 × 0.1 = **2e-5**，AdamW 归一化后单步更新量 ≈ lr = 2e-5。
4. **2e-5 < ULP → 更新被舍入成 0。909M 参数的 backbone 很可能根本没在训。**

而地图的整个配方建立在「backbone 解冻」上（论文 Table 3 的「3DGS frozen」消融：
冻住 T-mIoU 58.86 vs 联合训 64.03）。如果这个 bug 在，我们会**以为**自己在做全量微调，
实际跑出的是被消融证明更差的那一档，且从 loss 曲线上看不出来。

顺带澄清票面外的一个误解：Lightning `precision: bf16-mixed` **只包 autocast，不会额外维护
fp32 master weights**。所以「反正 mixed precision 会兜底」的想法不成立——这里参数本身就是 bf16。

## 要做的

1. **确认**这条推理（读 optimizer 构造路径，确认 aggregator 参数确实以 bf16 进 param group；
   确认没有别处把它转回 fp32）。**不要靠推理下结论**——本机没 GPU，但 dtype 是静态可查的。
2. 决定修法并说明理由。候选：
   - (a) 直接去掉 `.to(torch.bfloat16)`，参数留 fp32，靠 `bf16-mixed` 的 autocast 省激活显存。
     **倾向这个**：这是 PyTorch/Lightning 混合精度的标准姿势，且不引入新超参。代价是参数+
     优化器状态显存上升（R3 的分项表里已经把这笔算进去了，静态 17.0 GB 就是按此估的，
     所以**不会推翻 R3 的结论**）。
   - (b) 保留 bf16 参数但外挂 fp32 master weights。复杂，且等于自己实现一遍 AMP。
   - (c) 提高 transformer lr 让更新量超过 ULP。**不可接受**——偏离论文的 lr 配方去迁就一个 bug。
3. 查清这个 cast 是 AnySplat 上游原样继承的还是本仓库加的（`git log -L124,124:src/model/arch/anysplat.py`），
   如果是上游的，说明上游场景是**冻结 backbone**（那时 bf16 无害），我们解冻后它才变成 bug。
   这条要写进 `## 解决`，因为它解释了为什么既有实验没暴露这个问题。

## 硬约束

地图的「不干涉既有实现」：`config/model/encoder/anysplat.yaml` 的既有字段语义不动。
如果修法需要开关，用**新增的、默认保持现状**的可选字段，让既有实验的行为逐比特不变。
SegVGGT 那条线（`segvggt.py` 等）一律不碰。

## 验证（CPU）

- 构造 encoder 后打印 aggregator 各 param 的 dtype，确认为期望值。
- 构造 optimizer，取 transformer 组里一个参数，手动跑一步 `optimizer.step()`（用合成梯度，
  量级设成 2e-5），**断言参数确实变了**（`assert not torch.equal(before, after)`）。
  这个测试就是这张票的判定性证据，它在修复前应该**失败**。
- 确认既有 `instseg_anysplat.yaml` 路径的行为不变。

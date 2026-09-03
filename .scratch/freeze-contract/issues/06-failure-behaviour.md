# 06 — 校验失败时做什么

Type: grilling
Status: open
Blocked by: 03, 04

## Question

哪些不一致是**硬错**（`raise`，训练拒绝启动），哪些只是 warn，哪些只记录？

候选分档：

- **硬错**：指纹与 lock 不符；跨 stage 约束被违反；`freeze_keywords` 零命中（已有）。
- **待定**：可训参数里出现非 fp32 的（bf16 静默冻结的直接判据）——这在
  `instseg_anysplat`（`freeze_backbone: true`）下是良性的，因为 aggregator 根本不进
  优化器；一刀切会误伤。判据可能得是"**进了优化器**的参数里有非 fp32 的"。
- **只记录**：参数总量变化、lr 分组构成。

还要决：

1. **逃生门**：一定会有"我就是要改，别拦我"的时候。是 `--update-lock` 一次性重生成，
   还是 config 里一个 `skip_freeze_check: true`？后者会被滥用成永久开着。
2. **打印在哪**：rank 0 的 logger（`base_wrapper.py:529-532` 已有先例）、
   TensorBoard text、还是落一个文件进 run 目录？
3. **`sanity check` / `fast_dev_run` 下要不要也跑**。

## 完成判据

每一类不一致的处置定下来，逃生门的形态定下来。

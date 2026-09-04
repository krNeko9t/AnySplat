# 06 — 校验失败时做什么

Type: grilling
Status: open
Blocked by: 03, 04

## Question

哪些不一致是**硬错**（`raise`，训练拒绝启动），哪些只是 warn，哪些只记录？

候选分档：

- **硬错**：结构哈希与 lock 不符；跨 stage 约束被违反；`freeze_keywords` 零命中（已有）。
- **只记录**：`base_lr`、参数总量变化（进 lock 头部/正文，不参与判等）。

**03 号票已删掉一档**（2026-09-04）：原有的「可训参数里出现非 fp32 的就报错」整条撤销。
它自己就要一个「只算**进了优化器的**参数」的分支 if 来避开 `instseg_anysplat` 的误伤，
而那是在为另一层的另一个机制（bf16 永久 cast ⇒ AdamW 状态也是 bf16）擦屁股——**bf16 死参数
不是冻结**，它归 07 号票。`dtype` 在指纹里只是一列数据：变了哈希就变，人在 diff 里看见，
没有专门断言。

⇒ **本票的实际待决面因此收缩到只剩下面两问**（分档已定：硬错一档 + 只记录一档）。

还要决：

1. **逃生门**：一定会有"我就是要改，别拦我"的时候。是 `--update-lock` 一次性重生成，
   还是 config 里一个 `skip_freeze_check: true`？后者会被滥用成永久开着。
2. **打印在哪**：rank 0 的 logger（`base_wrapper.py:529-532` 已有先例）、
   TensorBoard text、还是落一个文件进 run 目录？
3. **`sanity check` / `fast_dev_run` 下要不要也跑**。

## 完成判据

每一类不一致的处置定下来，逃生门的形态定下来。

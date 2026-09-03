# 01 — 实测当前每份 config 的真实可训集合

Type: task (AFK)
Status: open
Blocked by: —

## Question

在设计指纹之前，先拿到**事实基线**：把所有设了 `freeze_keywords` 的 9 份 experiment
config 各建一次模型（`paper_repo` env，CPU 建模即可，无需 GPU / 不用跑 forward），
逐份产出：

- 每个 keyword 的命中参数数 / 参数量
- **实际** `requires_grad=True` 的参数清单（按顶层模块 rollup + 总参数量）
- 每个参数的 **dtype**（这是第二真相源，见地图 Notes 第 2 条）
- 该参数落进哪个 `param_groups` 分组、实际 lr multiplier

9 份：`segvggt_finetune_agnostic` / `segvggt_agnostic_phys_joint` / `segvggt_physgm` /
`segvggt_scannet` / `instseg_iggt` / `phys_iggt` / `phys_prop_iggt` / `physgm_iggt` /
`physgm_dpt_iggt`。

**这张票不做判断，只出数字。** 它是 03（指纹放什么）和 04（契约怎么表达）的前置——
没有真实指纹样本，格式设计就是凭空想象。

**顺带会自动暴露的东西**：把实测的可训集合与 `docs/repo_knowledge.md:118/126/148` 声称的
意图逐份对照，任何不一致就是一个现存的静默 bug。**发现不一致就记下来，不要顺手修**——
修法是 04/06 号票的事。

## 完成判据

一份 `.scratch/freeze-contract/notes/trainable_sets.md`，9 份 config 各一节，
每节含上述四项 + 与文档声称意图的一致性结论（一致 / 不一致 + 差在哪）。

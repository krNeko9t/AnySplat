# 01 — 要不要放开 LoRA（冻结底座 vs LoRA joint）

Type: grilling
Status: open
Blocked by: —

> **2026-09-03 改写**：本票原名"修复 freeze_keywords 误冻 LoRA（开训前必做）"，判定
> `segvggt_agnostic_phys_joint.yaml` 存在意图与实际相反的 bug。**该诊断已核实为错误**，
> 见下方"更正"。真问题是一个训练配方决策，本票据此重写。

## 更正：这里没有 bug

原诊断认为 config 名里的 "joint" 指 SegVGGT 论文 Table 7 第二行的 "LoRA joint"，因此
`freeze_keywords` 含 `frame_blocks`/`global_blocks` 把 LoRA 一起冻掉是与意图相反的 bug。

**"joint" 指的是"分割 + 物理同训"，不是 "LoRA joint"。** 代码行为 = config 注释 =
文档意图，三者一致：

- `config/experiment/segvggt_agnostic_phys_joint.yaml:2-3` — "Class-agnostic instance +
  per-query physics, **JOINTLY** trained on the same object queries (shared Hungarian match)"
- 同文件 `:52-55` — "Freeze aggregator + geometry (same spirit as stage-1) ...
  **Trainable: instance_ + semantic_head + query_physgm**"
- `config/experiment/segvggt_finetune_agnostic.yaml:17` — "global blocks **incl. their
  pretrained LoRA**) and the geometry heads stay frozen"
- `docs/repo_knowledge.md:118` — "整个 aggregator（**含 ckpt 里已训好的 LoRA**）+
  几何头 + bare tokens 全冻"
- `docs/repo_knowledge.md:126` — "冻结集同纠正后的 stage-1"

encoder config 的 `use_lora: true, lora_rank: 32` 是为了**建出 LoRA 模块好让官方 ckpt
的键能加载**（注释 "must match the trained ckpt: rank 32" 说的正是这个），不是为了训它。

⚠️ **原提的改法（在 `base_wrapper.py:512` 加 `lora` 排除）会同时静默改掉三份 config**
（stage-1 / joint / physgm）的既定语义——那才是引入 bug。**不要那样改。**

## Question

真问题：**这张图要不要放开 LoRA？**

- 冻结底座（现状）= SegVGGT 论文 Table 7 第一行；LoRA joint = 第二行。
  **23.4 → 31.9 mAP。⚠️ 这是论文数字，不是本仓库实测，不同数据/配方下不可直接套用。**
- 代价一：可训参数从 487M+0.2M 涨到 +9.44M LoRA（`lora_` 192 个参数 / 9.44M，实测见
  `docs/repo_knowledge.md`），显存与训练时间的账要重算。
- 代价二：底座一动，"几何不可能漂移"这条前提就没了，`segvggt_geo.weight: 0` 要重新考虑。
- 代价三：物性标注子集比分割数据小得多，放开底座在小数据上塌几何的风险实打实。

放开的**正确改法**（如果决定放开）：从这份 config 的 `freeze_keywords` 里去掉
`frame_blocks`/`global_blocks`，**并**补上块内 MLP/norm 的冻结——因为 LoRA 默认
`target_mlp=False`，直接删关键词会把 MLP 和 norm 一起解冻 = 接近全量微调。
`base_wrapper.py` 一个字都不用改。

**决策关联**：冻结机制本身的问题另开了一张图
（`.scratch/freeze-contract/map.md`），本票只管配方，不管机制。

## 完成判据

三选一定下来并写进 config：(a) 维持全冻底座；(b) 放开 LoRA（含 MLP/norm 的处置）；
(c) 先跑 (a) 出第一个 checkpoint，把 (b) 作为对照组押后。

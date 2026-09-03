# 01 — 实测当前每份 config 的真实可训集合

Type: task (AFK)
Status: closed
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

---

## 解决（2026-09-04）

九份 config 全部实测完毕，产出 `.scratch/freeze-contract/notes/trainable_sets.md`
（四项读数逐份齐全 + 与 `docs/repo_knowledge.md` 的一致性结论），逐参数原始记录在
`notes/raw/<experiment>.json` 的 `params` 字段（name → numel / dtype / requires_grad），
即 03 号票要设计的指纹的**样本数据**。工具：`scripts_probe.py` + `scripts_report.py`。

**七份一致，两处不一致**（只记录，未修）：

- **F1 · `segvggt_agnostic_phys_joint` 的实际 lr 比注释声称的高 5 倍。**
  `lr_multiplier: 5.0` 的注释假设 base lr = 2e-5，实际是 1e-4 ⇒ `query_physgm` 拿到
  5.00e-4（想要 1e-4），pretrained 侧 1.00e-4（自陈「same as corrected stage-1」= 2e-5）。
  可训集合完全正确、零命中护栏不响——**现有两道护栏结构上抓不到纯数值型偏差**。
- **F5 · `repo_knowledge.md:222` 关于 `freeze_module` 的一句与代码不符。**
  文档说它「仅在 freeze_keywords 为空时生效」，代码里没有这个条件：它在 `__init__`
  里无条件跑，`setup()` 在之后；且 `anysplat.py:191-193` 的无条件赋值会**解冻**，
  而 `freeze_keywords` 只冻不解冻，冻不回去。直接喂 02 号票的方案 2。

**另外五条记录**：F2 `register_token` 裸子串多命中 DINO 的 `patch_embed.register_tokens`
（今天良性，但证明 keyword 命中数不等于「冻对了」，rollup 粒度看不见、逐参数才看得见）；
F3 **bf16 静默冻结在这九份里零次发生**（IGGT 五份都把 bf16 的 aggregator 冻了），
06 号票那条 dtype 判据是纯前瞻护栏而非在修现存 bug；F4 `phys_iggt` 物理阶段仍训
`point_head`/`depth_head`（65.3M，显式进了 0.1 倍组，像是刻意），同链相邻配方给出相反答案
⇒ 交 05；F6 五份 IGGT 的 `backbone_lr_multiplier` 是死配置；F7 legacy 分支造空 param group
（行为无误）。

**顺带算出的 stage 链集合关系**（交 05 号票）：`trained(stage1) \ frozen(stage2)` = 0、
`trained(stage1) ∩ trained(stage2)` = 0、`trained(joint) ⊇ trained(stage1)` 成立。今天这条链自洽。

**环境代价**（本机 `paper_repo` env 被修改）：为跑通探针装了 `hydra-core lightning beartype
timm tabulate colorama wandb matplotlib plyfile trimesh lpips scikit-video colorspacious`；
`~/.cache/huggingface/hub/models--facebook--VGGT-1B` 留了一个 117M 的未完成分片
（本会话开始的下载，后来改走 `--random-backbone` 不再需要；HF 下次会续传）。

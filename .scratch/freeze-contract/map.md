# Map: 让"这个参数到底在不在训"变成可验收的事实

Label: wayfinder:map

## Destination

一层**可训参数指纹（trainable-parameter fingerprint）验收**，在训练启动时当场断言本次
run 的实际可训集合与配方声明一致；配套一份权威冻结契约文档。

**判据**：漏冻（如 `camera_token`）、误冻（如把本 stage 主体写进 freeze）、以及不走
`requires_grad` 的静默冻结（bf16 舍入），三类都在 step 0 炸掉，而不是训完才发现。

**不是**"设计一套新的冻结 API"——见 Out of scope 第一条。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）：终点是落地的校验层，不是它的
方案书。task 票直接动手；grilling 票只出决策。

**领域**：PyTorch Lightning + DDP 下的参数冻结与优化器分组。ICLR 2027 截止 **2026-09-25**。

**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

### 本图开工前已定的事（2026-09-03 grilling，不再重开）

1. **终点是验收，不是新语法**。核心痛点是"可训的 instance 分支长在 `Aggregator` 内部"
   （`src/model/segvggt/models/aggregator.py:182-234`），SegVGGT 路线因此**无法用一个
   `aggregator` handle 冻底座**，被迫逐子模块枚举——换任何声明式语法照样要枚举。
   会咬人的是"枚举漏了没人发现"，那是校验问题。
2. **指纹必须超出 `requires_grad`**。只查 `requires_grad` 是假的安全感：`cbe93f9`
   记录的 bf16 永久 cast 让 68.5%（≈623M）参数 50 步后从不更新，`requires_grad=True`
   全程为真，loss 曲线看不出来。dtype 是第二真相源。
3. **`freeze_keywords` 现有两道护栏保留**：零命中 `raise`（`base_wrapper.py:523-525`）、
   只冻不解冻（`:515-521`）。要补的是第三道：**声明的可训集合 vs 实际的可训集合**。
4. **官方实现保留 + 记录**：VGGT / SegVGGT / AnySplat vendored 代码里的冻结行为
   （LoRA 构造期冻基座、`mask_token` 冻结）不改，只写进契约文档。

### 事实底座（可复核，别重新推导，2026-09-03 现场读码）

- `freeze_research.md` — 6 种冻结实现的分类，带 file:line
- `docs/repo_knowledge.md:118 / :126 / :148 / :222` — 各 stage 冻结集的**意图**与实测参数量

关键事实：
- 唯一通用入口 `BaseWrapper.setup()`（`src/model/wrapper/base_wrapper.py:501-528`），
  匹配是裸子串 `kw in name`（`:512`）。
- `camera_token` = 1×2×1×1024 = **2048**，`register_token` = 1×2×4×1024 = **8192**，共
  **10240**（`aggregator.py:176-177`），`:486` 拼进每帧 token 序列喂给所有输出头。
- `freeze_backbone`/`freeze_module` **只存在于** `src/model/arch/anysplat.py:157-193`；
  `segvggt.py`/`iggt.py` 里 `grep requires_grad` **零命中**。用 `freeze_module` 的 5 份
  config 没有一份设 `freeze_keywords` ⇒ 两套今天按 arch 天然隔离，但无任何机制保证。
- `cbe93f9`（bf16 dtype 修复）**不在 `fix` 分支上**；`anysplat.py:124`、`iggt.py:80`
  仍是无条件 `.to(torch.bfloat16)`。`arch/segvggt.py` 干净（只用 autocast）。
- 冻结相关 config 注释共 14 行；告警型长注释集中在 `base_wrapper.py:502-518` +
  `repo_knowledge.md` 四段。
- 历史上 freeze 相关修复 commit 共 4 次：`12aaec6`（改为增量式）、`5f1eff7`
  （补 camera_token/register_token）、`7e196e9`（stage-1 误冻 instance 主体）、
  `cbe93f9`（bf16 静默冻结）。

## Decisions so far

<!-- 一行一个已关闭的票 -->

- [01 — 实测当前每份 config 的真实可训集合](issues/01-measure-current-trainable-sets.md)：
  九份配方的事实基线已落地（[读数](notes/trainable_sets.md) + [逐参数原始记录](notes/raw/)，
  后者即 03 号票要设计的指纹的样本数据）。七份与 `repo_knowledge.md` 一致；两处不一致：
  **`segvggt_agnostic_phys_joint` 的实际 lr 比注释高 5 倍**（可训集合完全正确，
  现有两道护栏抓不到纯数值型偏差 ⇒ 03 号票「指纹要不要带 lr」的实物论据），
  **`repo_knowledge.md:222` 关于 `freeze_module` 生效条件的一句是错的**（⇒ 02 号票）。
  另：**bf16 静默冻结在这九份里零次发生**（IGGT 五份都把 bf16 的 aggregator 冻了），
  06 号票的 dtype 判据是前瞻护栏而非现存 bug；`phys_iggt` 物理阶段仍训几何头 ⇒ 05 号票；
  stage 链集合关系已算出且今天自洽 ⇒ 05 号票。

## Not yet specified

- **长注释怎么收敛**：哪些告警注释在指纹层落地后变成冗余可删、哪些必须留。等指纹层的
  实际形态出来才判断得了。
- **新 arch 怎么被强制纳入**：将来加第四个 arch 时，指纹层是自动覆盖还是要手工接线。
- **`freeze_module` 的无条件赋值分支要不要彻底删**：依赖 AnySplat 路线是否还活着，
  以及 02 号票对"官方 vs 我们加的"的划线结果。

## Out of scope

- **把 `instance_*` 搬出 `Aggregator`**：物理上能让 backbone 变成可整体冻的干净模块，
  但要改 vendored 结构 + 全量 ckpt 键 remap + 与上游彻底分叉。距 ICLR 截止 22 天，纯风险。
- **"要不要放开 LoRA"（冻结底座 23.4 vs LoRA joint 31.9）**：这是训练配方决策，直接决定
  第一个 checkpoint 的质量，属于 `phys-on-query` 图的终点范围，不是冻结机制问题。
  见 `.scratch/phys-on-query/issues/01-unfreeze-lora.md`。
- **"判断某段是否需要前向"这类手动筛查/显存优化**：冻结的目的只有三条——不建计算图、
  不进优化器、权重不变。其余不在此列。
- **教师网 CPU 卸载 / `torch.no_grad` 推理路径**（`freeze_research.md` 方案 4/6）：
  是推理与显存管理，不是训练期冻结。

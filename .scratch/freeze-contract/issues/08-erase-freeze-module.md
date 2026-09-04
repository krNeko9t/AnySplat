# 08 — 彻底清除 `freeze_backbone` / `freeze_module`

Type: task (AFK)
Status: open
Blocked by: 02

## Question

02 号票判定：`freeze_backbone` / `freeze_module`（方案 2）是 `freeze_keywords`（方案 1）的
严格功能子集，且唯一的非子集部分（else 分支的无条件赋值解冻）今天零影响。**冻结入口必须唯一**
⇒ 整支清除，6 份 config 迁到 `freeze_keywords`。

按「不兼容旧代码 = 彻底清除」：**不留兼容层、不留 deprecated 警告、不留迁移说明**，
过去时只进 commit message。

### 要动的地方

1. **代码**：`src/model/arch/anysplat.py`
   - 删 `EncoderAnySplatCfg` 的 `freeze_backbone: bool`（`:~100`）与 `freeze_module: Literal[...]`（`:92-98`）两个字段
   - 删 `__init__` 里 `:157-193` 整块（`if self.freeze_backbone: ... else: ...`）
   - 删 `self.freeze_backbone = cfg.freeze_backbone` 赋值（`:~137` 对应位置）
   - **保留** `:147-155` 的 distill 块（冻 + 搬 CPU），那是 `distill` 特性不是冻结入口
2. **config**（6 份）：
   | config | 现状 | 迁移为 |
   |---|---|---|
   | `dl3dv` / `co3d` / `scannetpp` | `freeze_backbone: false` + `freeze_module: patch_embed` | `optimizer.freeze_keywords: [patch_embed]` |
   | `multi-dataset` | 同上 | 同上 |
   | `instseg_anysplat` | `freeze_backbone: true # ugly` | `optimizer.freeze_keywords: [aggregator, camera_head, depth_head]` |
   | `instseg_small` | 两者皆未设（默认 `"None"`） | **不设 `freeze_keywords`**（零行为） |
3. **文档**：`freeze_research.md` 的「方案 2」**整节删除**（连同它在方案 1 节里的交叉引用）；
   `docs/repo_knowledge.md:222` 的例外句**整句删除**，只留「唯一入口是 `freeze_keywords`」。
   两处目前带着 02 号票写下的「待删」标注，一并清掉。

### 完成判据（硬）

**逐参数实测等价**，不是"看着对"。用 01 号票的探针（`.scratch/freeze-contract/scripts_probe.py`
+ `scripts_report.py`，`paper_repo` env / CPU / `--random-backbone`）对 **6 份 anysplat config**
各跑两次——迁移前（HEAD）与迁移后——比对 `params` 字段：

- 每个参数的 `requires_grad` **逐个相等**（不是总数相等）
- 每个参数的 `dtype` 逐个相等
- `param_groups` 归属与 lr 不变

任一份不等即迁移写错，不得合并。

**特别当心**：`patch_embed` 是裸子串。01 号票 F2 记录它在 IGGT 上多命中 4 个
`part_head.window_{self,cross}_atten` 参数。AnySplat arch 无 `part_head`，但**这必须由实测证明，
不是由推理证明**——`instseg_anysplat` / `instseg_small` 带 instance head（`instance_feat_dim > 0`），
它们的模块清单与上游 4 份不同。同理 `camera_head`/`depth_head` 会被 `distill_camera_head`/
`distill_depth_head` 裸子串多命中（`instseg_anysplat` 是 `distill: false`，无此模块，但要实测确认）。

**另一个坑**：`freeze_keywords` 零命中会 `raise`（`base_wrapper.py:523-525`）。若某份 config 的
keyword 在该 arch 下真的零命中，说明翻译错了，别靠改 keyword 绕过。

实测报告落 `.scratch/freeze-contract/notes/anysplat_migration.md`，原始记录进 `notes/raw/`。

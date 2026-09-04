# 08 — 彻底清除 `freeze_backbone` / `freeze_module`

Type: task (AFK)
Status: closed (2026-09-04)
Blocked by: 02 (closed)
Assignee: krNeko9t (session 52f0bf61)

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

---

## 解决（2026-09-04）

**已清除，六份配方逐参数等价实测通过。** 报告 [`notes/anysplat_migration.md`](../notes/anysplat_migration.md)，
原始记录 `notes/raw/08-pre/` 与 `notes/raw/08-post/`（各 6 份，git 忽略）。

### 做了什么

1. **代码**（`src/model/arch/anysplat.py`，净 -50 行）：`EncoderAnySplatCfg` 的
   `freeze_backbone` / `freeze_module` 两个字段、`self.freeze_backbone = cfg.freeze_backbone`
   赋值、以及 `if self.freeze_backbone: ... else: ...` 整块全部删除。`if self.distill:` 块
   原样保留（distill 特性，不是冻结入口）。
2. **配方**（5 份改，1 份不动）：四份上游 → `optimizer.freeze_keywords: [patch_embed]`；
   `instseg_anysplat` → `[aggregator, camera_head, depth_head]`；`instseg_small` 不设（本就零行为）。
3. **文档**：`freeze_research.md` 的「方案 2」整节删除，余下小节**重编号**（3→2 … 6→5）、
   开头计数改 5 种/前 2 种——留一个 1,3,4,5,6 的空号等于留迁移说明。
   `docs/repo_knowledge.md:222` 的例外句整句删除，只剩「唯一入口是 `freeze_keywords`」。
   全仓 `grep freeze_backbone\|freeze_module` 现在只在 `.scratch/freeze-contract/` 的
   地图与票据里命中，即本图自身的历史记录。

### 完成判据的实测结果

六份配方各跑两次（HEAD vs 迁移后，`paper_repo` env / CPU / `--random-backbone`），
`requires_grad` / `dtype` / `numel` / 参数名集合**逐个**比对，加上 `param_groups` 归属与 `lr`：
**零差异**。迁移前后没有任何参数的 `requires_grad` 发生翻转——这正面证实了 02 号票的判断：
无条件赋值那半边今天零影响。

08 号票预警的两个裸子串坑**都未发生**，且是实测证明的：`patch_embed` 的 688 次命中全部落在
`aggregator.patch_embed.*` 与 `distill_aggregator.patch_embed.*` 内（IGGT 上的 `part_head`
多命中在 AnySplat 侧不存在）；`camera_head`/`depth_head` 的多命中风险只在 `distill: true` 时
存在，而 `instseg_anysplat` 是 `distill: false`。零命中 `raise` 六份都没触发。

### 现场发现（两条，都记进了报告）

1. **AnySplat 路线的构造期冻结不变量只有一个参数**：`aggregator.patch_embed.mask_token`。
   它同时被 `patch_embed` 关键词命中，靠「只冻不解冻」而无差别。02 号票立的
   「构造期冻结不变量」在 AnySplat 侧的全部实体就是这一个。
2. **探针漏了第二条 HF 下载路径** ⇒ 已在 `scripts_probe.py` 补上：
   `pretrained_weights: "hf:..."` 走 `init_anysplat_from_hf`
   （`src/model/arch/weight_loading.py:42`），它 = 构造 + `load_state_dict`，
   又一份约 5GB 下载、只改参数**值**。不短路它，`instseg_*` 在 `--random-backbone` 下
   依然联网（实测卡 22 分钟）。**这条直接约束 09 号票**：生成端要复用这条探针路径跑纯 CPU，
   必须同样短路它，否则 22 份 lock 的批量生成会挂在网络上。

### 对下游的约束

- → [09 号票](09-implement-lock-layer.md)：（a）生成端必须短路 `init_anysplat_from_hf`，
  理由同上；（b）本票删掉的是 `setup()` **之外**的最后一个冻结入口，
  04 D2「lock 全覆盖 + 缺 lock 硬错」现在真正只有一个真相源要锁。
- → [10 号票](10-converge-docs-and-comments.md)：`freeze_research.md` 现在是 5 节、
  已无「待删」标注；10 号票整份删除时少一节要吸收。术语表里
  「构造期冻结不变量」的 AnySplat 实例可直接写 `mask_token` 一个。

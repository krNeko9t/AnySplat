# 03 — 可训参数指纹里放什么

Type: grilling
Status: closed
Blocked by: 01
Assignee: krNeko9t

## Question

**本图的核心决策。** 指纹要能同时抓住三类失败：漏冻、误冻、以及不走 `requires_grad`
的静默冻结（bf16）。

待决：

1. **粒度**：逐参数（几千行）/ 按顶层模块 rollup（十几行）/ 两级（rollup 给人看，
   逐参数哈希给机器比）？
2. **维度**：`requires_grad` 是必须的。**dtype 也是**（地图 Notes 第 2 条）。还要不要
   `param_groups` 归属 + 实际 lr？参数量？shape？
3. **形态**：人可读的清单（能进 code review、能 diff）还是一个哈希？还是两者都要——
   哈希做快速判等，清单做"差在哪"的定位？
4. **稳定性**：指纹必须对什么不敏感？参数遍历顺序、DDP rank、`num_semantic_classes`
   这类会改 shape 的 config 项——哪些进指纹哪些不进，决定它是好用还是天天误报。

**约束**：指纹必须在 `BaseWrapper.setup()` **之后、strategy wrap 之前**取得
（`base_wrapper.py:502-505` 的原因同样适用）。

## 完成判据

指纹的字段集合、粒度、序列化形态定下来，且能解释它为什么抓得住上述三类失败各一个
具体历史案例（`5f1eff7` 漏冻 token、`7e196e9` 误冻主体、`cbe93f9` bf16）。

---

## 解决（2026-09-04）

### 指纹的定义

**一个捕获点**：`BaseWrapper.setup()` 末尾（`base_wrapper.py:501-532`），wrap 之前。
**覆盖全量参数**，不只可训子集。**一行一参数，按 name 排序**，顶部一个哈希。

```
# freeze-lock: segvggt_physgm
arch: SegVGGTWrapper
freeze_keywords: [patch_embed, frame_blocks, ..., camera_token, register_token]
base_lr: 1e-4                      # 记录，不参与哈希、不判等
structure: sha256:a3f1...

- fp32   2048  model.encoder.model.aggregator.camera_token
- fp32   8192  model.encoder.model.aggregator.register_token
T fp32  66754  model.encoder.query_physgm.decoders.0.0.weight
```

- **字段**：`requires_grad`（`T`/`-`）、`dtype`、`numel`、`name`。**不含 `shape`**（`numel`
  已足以判「这个参数变了」）、**不含权重数值**。
- **头部**：`experiment` / `arch` / `freeze_keywords` 原文 / `base_lr` / 哈希。
  **不记 git commit**——它每次重生成都产生一行无关 diff，且会诱人当成「上次验过的版本」来信任。
- **不变量**：按 name 排序消除遍历顺序；`setup()` 时未 wrap，无 `module.` 前缀；
  **每个 rank 各自算、各自比对**（纯 CPU 遍历，成本可忽略；rank 间冻结不一致是最难从 loss
  曲线看出来的一类故障，白送的覆盖不放掉）。

**零个 if。** 校验就是「重算一份、和 lock 逐行比」。

### 为什么是全量而不是可训子集

三类失败确实全落在可训子集内（漏冻=多了、误冻=少了、dtype 变化=可训参数的列变了）。
选全量的理由只有一条：**可训子集分不清「被冻了」和「不存在了」**——arch 重构删掉一个模块，
与把它冻上，在可训子集里同形。实测规模（参数个数）：最小的一份可训只有 15
（`segvggt_physgm` / `physgm_iggt`），最大的 2070（`segvggt_scannet`），全量 1595–2621。
即使全量，也就是 `package-lock.json` 量级的机器生成文件。

### 为什么是定宽文本而不是 JSON

lock 唯一的读者场景是 `git diff`。JSON 每个参数占 3–4 行带大括号，一个参数的变化在 diff 里
散成一片；定宽文本一个参数恰好一行——`5f1eff7` 那类故障在 diff 里就是**一行**：
`camera_token` 从 `-` 变 `T`。JSON 留给 `notes/raw/` 的探针原始记录（那是给脚本吃的）。

### 三个历史案例的覆盖（完成判据）

| commit | 故障 | lock diff 里长什么样 |
|---|---|---|
| `5f1eff7` | 漏冻 `camera_token`/`register_token` | 两行的首列 `-` → `T` |
| `7e196e9` | stage-1 误冻 instance 主体 | `instance_*` 那批行的首列 `T` → `-` |
| `cbe93f9` | bf16 永久 cast | aggregator 那批行的 dtype 列 `fp32` → `bf16` |

### 砍掉的两样东西（本票的主要产出）

**1. `lr` / `param_groups` 判等——整个砍出这张图。**

`configure_optimizers`（`base_wrapper.py:529-598`）在 strategy wrap **之后**才跑，分组与 lr
在那里才成立。要把它们纳入判等，就得两点捕获 + 双哈希（结构半硬错 / lr 半可 warn）+
把 lr 存成 `(group_index, lr_multiplier, base_lr, effective_lr)` 四元——**round 2 提出的结构性
复杂度全部来自这里，一条也不来自 dtype**。

砍的判据是地图 Out of scope 自己写下的那条：**冻结的目的只有三条——不建计算图、不进优化器、
权重不变**。`lr` 三条都不占。F1（`segvggt_agnostic_phys_joint` 实际 lr 比注释高 5 倍）的真身是
「注释里假设的 base lr 过期了」，那是配方审查问题，不该由冻结层持枪站岗。

**保留可见性、扔掉执法**：`base_lr` 记进 lock 头部，不参与哈希、不炸。改了 lr，下次重生成
lock 时 diff 里红一行，人看得见。零分支、零误报、零逃生门。

**2. dtype 的专门断言——整条撤销，`dtype` 降级为数据列。**

06 号票原有一条待定判据「可训参数里出现非 fp32 的就报错」，它自己就写着一刀切会误伤
`instseg_anysplat`，得改成「**进了优化器的**参数里有非 fp32 的」——那正是一个分支 if，
而且是为**另一层的另一个机制**擦屁股。

`dtype` 留作一列：进指纹、进哈希、**不设任何断言**。边际成本是零（探针 `params` 字段早就产出
`[numel, dtype, requires_grad]`），而它保住了 dtype 变化的可见性——将来真有人给 aggregator
打开 bf16，哈希会撞一次，人被迫看一眼。

### 领域判据：「静默冻结」是比喻，不是同一个概念

bf16 死参数只占冻结三条判据中的第三条（权重不变），且是**另一层的另一个机制**：永久
`.to(bfloat16)` ⇒ AdamW 的 `exp_avg`/`exp_avg_sq` 也是 bf16 ⇒ ULP 吃掉更新。它有自己的开关
（`aggregator_param_dtype`）、自己的验证脚本（`T11_verify.py`）、自己的票（07）。管它叫
「静默冻结」这个比喻把它偷渡进了本图——**与 02 号票发现的「两个冻结入口」是同一类错误，
只是跑在概念层而非代码层**。

据此收窄地图 Notes 第 2 条：`dtype` 是**记录维度**，不是独立判据。

### 对下游票的约束输入

- **→ 04**：lock 的**正文形态与字段集合已定**（上），04 只决定**存放位置、生成与更新流程、
  以及 lock 是否强制**（02 号票要求它必须是强制）。头部含 `freeze_keywords` 原文这一点是给 04 的：
  它让 diff 里「改了哪个 keyword」与「因此哪些参数动了」上下并排。
- **→ 06**：**dtype 判据那一条整条删除**。失败行为只剩一档：结构哈希不符 = 硬错。
  06 的实际待决面因此收缩到只剩逃生门形态与打印位置。
- **→ 07**：与本图**脱钩**。07 是独立的洞，不必等这层落地；本层只保证它将来发生时 lock 会撞一次。
- **→ 04（可选）**：「冻结 = 不建计算图 / 不进优化器 / 权重不变」与「bf16 死参数不是冻结」
  这两条术语判定，应在契约文档里有一节。本仓无 `CONTEXT.md`，是否新建由 04 决定。

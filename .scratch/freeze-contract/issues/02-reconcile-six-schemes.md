# 02 — 六种冻结实现：哪些留、哪些记录、哪些删

Type: grilling
Status: closed
Blocked by: 01
Assignee: krNeko9t

## Question

`freeze_research.md` 分类出 6 种实现。给每一种定性，并划一条**"官方 vs 我们加的"**的线：

| # | 实现 | 位置 | 待定 |
|---|---|---|---|
| 1 | `freeze_keywords` + `BaseWrapper.setup()` | `base_wrapper.py:501-528` | 唯一入口，留 |
| 2 | `freeze_backbone` / `freeze_module` | `anysplat.py:157-193` | **是我们加的还是 AnySplat 官方的？** 无条件赋值分支（`:191-193`）语义与方案 1 相反 |
| 3 | LoRA 构造期冻基座 | `lora.py:123-125` | SegVGGT 官方，留 + 记录 |
| 4 | 教师网 `no_grad` + CPU 卸载 | — | 地图已判 out of scope |
| 5 | `mask_token.requires_grad_(False)` | `aggregator.py:302-304` | VGGT 官方，留 + 记录 |
| 6 | 推理期 `eval()` + 全冻 | — | 地图已判 out of scope |

真正要决的只有**方案 2**。已核实的事实（地图 Notes）：它只存在于 `anysplat.py`，
`segvggt.py`/`iggt.py` 零命中；用它的 5 份 config 无一设 `freeze_keywords`
⇒ **今天没有实际冲突，但没有任何机制保证明天也没有**。

选项：(a) 留着 + 让指纹层顺带把它的效果也算进去；(b) 留着但加一条启动期断言
"`freeze_keywords` 非空时 `freeze_module` 必须为 None"（把今天的隐式隔离变显式）；
(c) 判定为我们加的私货，删掉无条件赋值分支、把 5 份 config 迁到 `freeze_keywords`。

**先查 AnySplat 上游仓库确认方案 2 是不是官方代码**——这决定 (c) 是否违反地图
Notes 第 4 条。

## 完成判据

方案 2 的处置定下来；`freeze_research.md` 升级成契约文档的"现状"一节，每种实现标注
官方/自研 + 保留/删除。

---

## 解决（2026-09-04）

### 六种实现的定性

| # | 实现 | 位置 | 官方/自研 | 处置 |
|---|---|---|---|---|
| 1 | `freeze_keywords` + `BaseWrapper.setup()` | `base_wrapper.py:501-528` | 自研（`20aa5f0`） | **留**——唯一冻结入口 |
| 2 | `freeze_backbone` / `freeze_module` | `anysplat.py:157-193` | **AnySplat 官方**（`8d6180e`，逐字未改） | **彻底清除** → 08 号票 |
| 3 | LoRA 构造期冻基座 | `lora.py:123-125` | SegVGGT 官方 | 留 + 记录（构造期不变量，非入口） |
| 4 | 教师网 `no_grad` + CPU 卸载 | — | — | 地图 out of scope |
| 5 | `mask_token.requires_grad_(False)` | `vggt/aggregator.py:184-185`、`segvggt/aggregator.py:303-304` | VGGT 官方 | 留 + 记录（同上） |
| 6 | 推理期 `eval()` + 全冻 | — | — | 地图 out of scope |

### 决策：方案 2 彻底清除，并收窄地图 Notes 第 4 条

原地图 Notes 第 4 条「官方实现保留 + 记录」**被推翻一半**。立下的划线判据（领域词）：

> **冻结入口（freeze entrypoint）= config 能拨动的冻结开关。入口必须唯一，只有 `freeze_keywords`。**
> vendored 模型内部的**构造期冻结不变量**（方案 3 LoRA 基座、方案 5 `mask_token`）不是入口——
> config 拨不动，且原理上无法用 `freeze_keywords` 表达（`setup()` 跑时 LoRA 早已建完）——**保留 + 记录**。

方案 2 是纯 config 开关 ⇒ 落在入口这边 ⇒ 清除。「它是官方的」不再构成保留理由：真正的成本是**两个入口并存本身**，而对 AnySplat 上游意图的心智模型是空的，留着就是留一颗读不懂的雷。

**「删了以后 rebase 上游变难」是空的**：`ed8a9f4` / `16521cb` / `e2db80d` 三次重构已改掉目录结构与层间契约，与上游早已彻底分叉。

### 支撑事实（本会话现场核实）

**A. 方案 2 是 AnySplat 官方上游代码，且用它的 4 份 config 也是。**
`git log -S` 追到 release commit `8d6180e`（2025-06-30，原路径 `src/model/encoder/anysplat.py:160-195`）。
与今天 `arch/anysplat.py:157-193` diff：**唯一差异是一处 cosmetic 变量提取**（`modules_to_freeze`）。
`co3d` / `dl3dv` / `multi-dataset` / `scannetpp` 四份 config 同出 `8d6180e`；
`instseg_anysplat.yaml`（`fd107b5`，我们的）用的是 `freeze_backbone: true`，**没碰 `freeze_module`**。

**B. 方案 2 是方案 1 的严格功能子集。** 逐档翻译：

| 现状 | 等价 `freeze_keywords` | 使用者 |
|---|---|---|
| `freeze_backbone: true`（`pred_head_type=depth`） | `[aggregator, camera_head, depth_head]` | `instseg_anysplat` |
| `freeze_module: patch_embed`（+`distill: true`） | `[patch_embed]` | `dl3dv` / `co3d` / `scannetpp` / `multi-dataset` |
| `freeze_module: "None"`（dataclass 默认） | 不设——`setup()` `:508-509` 早返回，零行为 | `instseg_small` |
| `"all"` / `module_pairs` 三档 | `[aggregator]` / `[patch_embed, frame]` … | **零 config 使用（死代码）** |

`distill` 那半冗余：distill 块 `:147-155` 已冻过并搬去 CPU。

**C. 唯一的非子集部分就是那个 bug，且今天零影响。** else 分支（`:191-193`）的无条件赋值只能解冻
**比 `__init__` 更早**冻的东西 = `patch_embed.mask_token` + distill 块。两者名字恰好含
`patch_embed` / `distill` ⇒ 侥幸未被解冻 ⇒ **删掉它不改变任何现有行为**。

（连带纠正一条被两份文档写反的因果：并非「只在 `freeze_keywords` 为空时生效」。真实顺序
`__init__`（方案 2）→ `setup()`（方案 1），后跑的方案 1 只冻不解冻，**它赢**，净效果是并集。
`freeze_research.md:37` 与 `docs/repo_knowledge.md:222` 已在本票修正。）

**D. 迁移不能靠肉眼。** `patch_embed` 裸子串在 IGGT 上多命中 4 个
`part_head.window_{self,cross}_atten` 参数（01 号票 F2）。AnySplat arch 无 `part_head`，
但这决定了 08 号票的完成判据必须是**实测逐参数相等**，而不是"看着对"。

### 本票交付

- 六种实现的定性表（上）。
- 修正两处已证伪的文档：`freeze_research.md:37`（改写因果 + 标注整节待删）、
  `docs/repo_knowledge.md:222`（同）。

**未做**（Q4 显式划出）：「`freeze_research.md` 升级成契约文档的现状一节」——契约文档存哪、
长什么样是 04 号票的决定，在这里做等于替 04 做主。改为 04 的输入。

### 对下游票的约束输入

- **→ 04**：本票选「指纹层天然覆盖方案 2」而**不加**「`freeze_keywords` 非空 ⇒ `freeze_module` 必须为 None」
  那条专门断言，前提是 **04 把 lock 定成强制**。若 04 定成 lock 可选，需回头补一道等价护栏。
  （方案 2 删除后此约束自动失效，但 04 若在 08 落地前决策，仍需按此考虑。）
- **→ 04**：契约文档的「现状」一节由 04 决定形态；方案 2 那一节在 08 落地时**整节删除**，
  不留迁移说明、不留 deprecated 警告。

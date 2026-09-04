# 04 — 契约存哪、怎么写、谁维护

Type: grilling
Status: closed
Blocked by: 03
Assignee: krNeko9t

## Question

指纹要和**声明**比对才有意义。声明放哪：

(a) **写进 experiment yaml**：如 `optimizer.expect_trainable: {instance_: 454M, ...}`。
    改配方就要改声明，改动可见于同一份 diff。缺点：yaml 变长，且手写数字会过期。
(b) **旁挂 lock 文件**：`config/experiment/xxx.freeze.lock`，由脚本生成、提交进 git。
    改配方 ⇒ lock 不匹配 ⇒ 训练拒绝启动 ⇒ 人跑一次 `--update-lock` 显式确认。
    像 `package-lock.json`。缺点：多一类文件、多一个仪式。
(c) **不存声明，只做跨 stage 比对**：见 05 号票。适用于 stage 链，管不住单份配方写错。

还要决：**首次生成**怎么来？01 号票的实测产物直接落成初始 lock，还是人手写一遍？

**注意**：这条与地图 Out of scope 里"不设计新冻结 API"不冲突——`freeze_keywords`
的写法一个字不改，加的是它旁边的验收物。

## 完成判据

声明的存放位置、格式、生成与更新流程定下来，且明确"改配方"时人要做的动作是什么。

---

## 解决（2026-09-04）

四问一次定完，无二轮。

### D1 · 身份键 = hydra 的 experiment choice，一份 experiment 恰好一份 lock

lock 落 **`config/experiment/<X>.freeze.lock`**，`<X>` 就是 `+experiment=<X>` 里的那个名字
（= yaml 文件名）。与配方同目录同名 ⇒ 改配方与改 lock 落在同一次 `git diff` 里并排，正是
03 号票把 `freeze_keywords` 原文写进 lock 头部时想要的那个并排。

**`wandb.name` 被实测证伪，不可作身份键**（本票现场核对 22 份）：4 份与文件名不等
（`multi-dataset`→`multidataset-16gpu`、`scannetpp`→`vggt-mdataset-new-scannetpp-dynamic_batchsampler`、
两个 `_mv` 变体各自指向非 `_mv` 名），且**两组重名**——`instseg_inscene_infinigen` 与
`instseg_inscene_scannetpp_v2` 各被两份文件用。身份键重名 = 两份配方共用一份 lock，护栏当场失效。

**不按结构维度分岔。** `num_semantic_classes` / `instance_feat_dim` 从 CLI 覆盖会改 `numel`
⇒ 哈希必撞，这是**特性不是缺陷**：lock 的全部价值就是让这种改动必须被人签一次字。给它开
一条按维度分岔的旁路，等于把逃生门做成默认路径。

### D2 · 覆盖面 = 22 份全覆盖，**缺 lock = 启动硬错**

不是「有 lock 就校验、没有就跳过」。包括今天不设 `freeze_keywords` 的 13 份。

两条理由：

1. **02 号票的删除决策挂在这个前提上。** 它选「指纹层天然覆盖 `freeze_module`」而不加专门断言，
   写明前提是「04 把 lock 定成强制」。若定成可选，那个前提降级为君子协定，02 就得回头补护栏。
2. **它把第三种历史故障形态也纳入：忘了写 `freeze_keywords`。** 与 03 号票选全量参数而非可训
   子集是同一条理由——**不能让「没有」和「被冻了」同形**。lock 可选时，「这份配方没有 lock」与
   「这份配方不需要冻结」同形。

代价（认下）：13 份无冻结需求的配方各背一个 lock 文件。它们的 lock 全是 `T` 列，diff 里安静。

### D3 · 生成端 = 独立 CPU 脚本，**capture 函数唯一**

`scripts/freeze_lock.py +experiment=<X>`，复用 01 号票已跑通的探针路径
（hydra compose → `load_typed_root_config` → 建模 → `setup()`），`--random-backbone` 免 HF 权重。

**不走 `src/main.py ++freeze.update_lock=true`**：那要起一遍 Trainer 才能到 `setup()`，把一件
纯 CPU 的事绑死在 GPU 节点上。01 号票已实测九份全部可在无卡机器上建模。

**硬约束（比选哪个脚本更重要）**：生成端与 `setup()` 末尾的校验端**必须调用同一个 capture 函数**。
两端各写一份序列化逻辑，护栏就在守自己的影子。

### D4 · 文档三分：术语进新建 `CONTEXT.md`，现状进 `repo_knowledge.md`，`freeze_research.md` 整份删

- **新建根 `CONTEXT.md`，只放术语**（本仓原本没有）：冻结三判据（不建计算图 / 不进优化器 /
  权重不变）、**冻结入口** vs **构造期冻结不变量**（02 立的词）、**指纹**、**lock**、以及
  「**bf16 死参数不是冻结**」（03 立的词）。**只放词，不放实现**。
- **机制现状**（六种实现的定性表、唯一入口、lock 生成与更新流程）并进 `docs/repo_knowledge.md`。
- **`freeze_research.md` 在内容被吸收后整份删除**，按「不兼容旧代码 = 彻底清除」，不留迁移说明。

选新建 `CONTEXT.md` 而不是把术语塞进专题文档，理由是：那两条术语判定的用途**根本不在冻结这一件
事上**。02（代码层留了第二个入口）与 03（概念层把 bf16 偷渡进来）是**同一个错误跑在两层**，
术语表正是防这类错的常驻器官——埋进 `freeze_contract.md` 就只有查冻结的人会读到它。

### 「改配方时人要做的动作」（完成判据的正面回答）

1. 改 `config/experiment/<X>.yaml`；
2. 启动 ⇒ `setup()` 末尾结构哈希与 lock 不符 ⇒ 硬错，训练拒绝启动；
3. 跑一次 `python scripts/freeze_lock.py +experiment=<X>` 重生成 lock；
4. **在 diff 里看一眼哪些参数的首列/dtype 列变了**——这一眼就是签字；
5. yaml 与 lock 一起进同一个 commit。

### 顺带关掉一块雾（原 Not yet specified 第二条：新 arch 怎么被强制纳入）

D2 已经回答了：**自动覆盖，无需手工接线**。新 arch 总是以新 experiment config 的形式到来，
而缺 lock 是硬错 ⇒ 它跑不起来，直到有人生成一份 lock 并看过它。该条从雾里删除。

### 对下游票的约束输入

- **→ 06**：失败行为的分档已被 D2 追加一条**硬错**——「lock 文件缺失」。06 只剩逃生门形态
  （`--update-lock` 已由 D3 定成独立脚本 ⇒ 逃生门天然是「重生成并提交」，`skip_freeze_check`
  那个会被滥用成永久开着的开关**不必再考虑**）与打印位置、`fast_dev_run` 三问。
- **→ 09**（新票，实现）：capture 函数唯一是硬约束；22 份配方的初次 lock 批量生成是完成判据。
  **已知风险**：01 号票只证明了 9 份能在 CPU/无数据集下建模，剩余 13 份未验；若某份建不起来，
  是那份配方本身的问题，按 D2 它本来就跑不起来。
- **→ 10**（新票，文档）：D4 的三分方案是它的规格。

# 10 — 文档三分与长注释归位

Type: task
Status: closed (2026-09-04)
Blocked by: 09 (closed 2026-09-04)
Assignee: krNeko9t (session 59348cce)

## Question

04 D4 定下的三分方案的落地票。lock 层落地（09）之后做——注释能不能删，取决于指纹层是否真的在守着。

1. **新建根 `CONTEXT.md`，只放术语，不放实现**：
   - 冻结的三判据：不建计算图 / 不进优化器 / 权重不变
   - **冻结入口**（config 能拨动的冻结开关，唯一，只有 `freeze_keywords`）
     vs **构造期冻结不变量**（LoRA 冻基座、`mask_token`；config 拨不动，保留 + 记录）——02 号票立的词
   - **指纹**（全量参数的结构快照）、**lock**（提交进 git 的那份指纹）
   - **bf16 死参数不是冻结**（只占第三条判据「权重不变」，机制在另一层：永久 cast ⇒ AdamW
     状态也是 bf16 ⇒ 量级 ~lr 的更新被舍成 0，而 `requires_grad` 全程为真）——03 号票立的词。
     **词条到此为止，不指向任何票**：07 已整票判出 scope（属训练精度策略），而 `CONTEXT.md`
     是常驻文件、`.scratch/` 会被删，常驻文档不得指向 scratch 路径。要写指向就指向
     `anysplat.py:113` / `iggt.py:80` 这两行代码
2. **机制现状并进 `docs/repo_knowledge.md`**：02 号票的六种实现定性表、唯一入口、
   04 的 lock 生成与更新流程（「改配方时人要做的五步」）。
3. **`freeze_research.md` 整份删除**（内容被 1/2 吸收后）。按「不兼容旧代码 = 彻底清除」，
   不留迁移说明、不留 deprecated 标注——它目前带着 02 号票写下的「待删」标注，一并清掉。
4. **长注释收敛**（原地图 Not yet specified 第一条）：冻结相关 config 注释共 14 行，
   告警型长注释集中在 `base_wrapper.py:502-518` + `repo_knowledge.md` 四段。逐条判定：
   指纹层落地后哪些变成冗余可删、哪些必须留。
   **判据**：一条注释若只是在提醒「别忘了冻 X」，而 lock 现在会为它硬错 ⇒ 删；
   若解释的是**机制为什么必须这样**（如 `:502-505` 那段「setup 在 wrap 之前」的因果）⇒ 留。

## 完成判据

`CONTEXT.md` 存在且只含术语；`freeze_research.md` 不存在；14 行注释逐条有去处（删/留 + 理由）。

---

## 05 号票 D2 的输入（2026-09-04）：`CONTEXT.md` 追加一个隔离词

**权重落位断言 ≠ 指纹**。`_assert_query_physgm_coverage`
（`src/model/arch/segvggt.py:297-320`）查的是「ckpt 里有 `query_physgm` 权重就必须全部
加载成功」——属于**权重的值是否落位**，冻结三判据（不建计算图 / 不进优化器 / 权重不变）
**一条都不占**。它留在原地，一行不改，也不并进指纹层。

立这个词的理由与 04 立术语表同源：把它并进指纹层，和 02 号票清除的 `freeze_module`
是**同一类错误的两面**——那次是把不同机制混进一个入口，这次是把不同判据混进一个层。


---

## 决议（2026-09-04）

**三分方案已全部落地，`freeze_research.md` 已删（`git rm`），20 行 config 冻结注释逐条判完。**

### 1. 根 `CONTEXT.md` 已建（50 行，纯术语）

七个词条，一条实现细节都不放：**冻结**（三判据）、**冻结入口**（唯一，`freeze_keywords`）、
**构造期冻结不变量**、**指纹**（全量而非可训子集，附理由）、**lock**（被执法的是正文不是它自己的哈希）、
**bf16 死参数不是冻结**、**权重落位断言 ≠ 指纹**（05 D2 的输入）。每条带 `_Avoid_`；
按票面要求，bf16 词条**只指向 `arch/anysplat.py:113` / `arch/iggt.py:80` 两行代码**，
不指向任何票、不出现 `.scratch/` 路径。`_assert_query_physgm_coverage` 的锚点已现场核对（`segvggt.py:297`）。

### 2. 机制并进 `docs/repo_knowledge.md`，新增 §6（六小节，原 §6/7/8 顺延为 §7/8/9）

`6.1` 唯一入口的三条语义（裸子串 / 只冻不解冻 / 零命中 raise）；`6.2` **其余四处
`requires_grad=False` 逐条定性的表**（LoRA、`mask_token`、教师网 CPU 卸载、推理期 eval），
外加一句"历史上存在过第五处 `freeze_module`，已彻底删除"；`6.3` lock 的身份键 / 全覆盖 /
时机 / 硬错三条 / 只记录两项 / 比正文不比哈希 / 失败落盘 / 每 rank 各算；
`6.4` **改配方五步**（含 `--all` 是串行子进程 + 单份 7.3G 的理由）；`6.5` 四条不由这层负责的事
（`lr`、跨 stage、bf16、权重落位）——把三张出图票的结论钉在常驻文档里，防它们被当成"还没做"再捡回来。

§8（原 §7）第 5 条压缩成指针（"判断某参数是否在训**看它的 lock**，不看注释"）；文件头加一行
"术语以根 `CONTEXT.md` 为准"。

**09 号票预警的"至少一条内容错误"找到了，是 §8 第 6 条**：原文写"VGGT aggregator 跑 bf16 autocast"，
但 `arch/anysplat.py:113` / `arch/iggt.py:80` 是**无条件的永久 `.to(torch.bfloat16)`**，不是 autocast——
这正是 bf16 死参数的机制，被一句"autocast"盖住了。已改写：点名两行代码、写出 AdamW 状态也是 bf16
的后果、点明**这不是冻结**并指回 `CONTEXT.md`、带上 5/22 与 909M 的命中面。
（这条错误能活这么久，恰好是 04 立术语表的论据：概念混淆先在文档里发生。）

### 3. `freeze_research.md` 整份删除

内容已被 1/2 吸收。按「不兼容旧代码 = 彻底清除」：无迁移说明、无 deprecated 标注、无重定向存根。
全仓（`.scratch/` 外）对它零引用，核对过。

### 4. 20 行 config 冻结注释的逐条去处

**票面写的是 14 行，实测是 20 行（4 份配方）**——14 是 02 号票时期的读数，此后 08 号票的迁移
与 physgm 的注释增补都没进那个计数。判据照 D4 用（只提醒"别忘了冻 X" ⇒ 删；解释"机制为什么必须这样" ⇒ 留）：

| 位置 | 原 | 现 | 判定 |
|---|---|---|---|
| `segvggt_physgm.yaml` 冻结块 | 12 | 9 | **删 6 留 6 改写**。删掉 `camera_token/register_token` 那 5 行警告（"easy to miss"、"physics-only 目标会悄悄拖动它们"、"没有东西拉回来"）——`5f1eff7` 那类漏冻现在在 lock diff 里就是多出的两行，硬错；只留一句**为什么**它们在冻结集里（喂每一个输出头）。删掉 `-> verified: 只有 query_physgm 训（~0.2M）` ——手抄的实测数，lock 现在是权威且不会过期。留下并合并两条**裸子串匹配语义**（`instance_` 一词覆盖四个子模块；`query_physgm` 挂在 arch 上所以下面任何关键词都够不着它）——这是机制。 |
| `segvggt_agnostic_phys_joint.yaml` | 4 | 3 | **删 1**：`# Trainable: instance_ + semantic_head + query_physgm.` 是 lock 正文的手抄摘要。留 3 行意图（为什么冻 aggregator+geo）。 |
| `segvggt_finetune_agnostic.yaml` | 3 | 3 | **改写不删**：`semantic_head / instance_ must stay OUT of this list` 是 `7e196e9`（误冻本 stage 主体）的告警，lock 现在会硬错 ⇒ 去掉祈使语气与手抄的 `~487M`，只留陈述"它们正是本 stage 要调的"。 |
| `segvggt_scannet.yaml:40` | 1 | 1 | **留**：`# DINO backbone frozen (paper)` 是**出处**（论文 A.4），lock 不承载出处。 |
| `segvggt_physgm.yaml` 头部 3 行（"要改成联合训就把 X 移出 freeze_keywords"） | 3 | 3 | **留**：配方使用说明，不是冻结提醒。 |

`repo_knowledge.md` 四段同样过了一遍：`:118` 删掉 `**切勿**把 semantic_head/instance_ 写进 freeze`
（同上，lock 硬错）；`:120` 的 LoRA 覆盖面收成指向 §6.2 的一行（机制只该有一处权威）；
`:126`/`:148` 是配方意图与裸子串语义，**留**。

`base_wrapper.py` 的长注释（`apply_freeze` 与 `setup` 的 docstring）**一行不动**：讲的是
"setup 在 DDP wrap 之前所以冻结与校验都必须在这里"、"只冻不解冻否则 LoRA 失效"、
"拆出 apply_freeze 否则生成一份 lock 需要先有那份 lock"——三条全是机制因果，D4 判留。

### 5. 实测

`segvggt_physgm` / `segvggt_agnostic_phys_joint` / `segvggt_finetune_agnostic` 三份逐一重跑
`scripts/freeze_lock.py +experiment=<X>`，三份都报 `unchanged` ⇒ 注释改写对指纹零影响。

### 对下游

无。本图无后继票。

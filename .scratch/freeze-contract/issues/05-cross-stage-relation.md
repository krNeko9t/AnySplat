# 05 — 跨 stage 的冻结关系怎么表达

Type: grilling
Status: closed (out of scope)
Blocked by: 03
Assignee: krNeko9t

## Question

单份配方自洽还不够。真正致命的是**stage 链断裂**：stage-2（`segvggt_physgm`）声称
"几何完全不动，所以这份 ckpt 应当复现 stage-1 的几何"，靠的是冻结集**覆盖了**
stage-1 训过的一切。`camera_token`/`register_token` 这 10240 个参数漏一个，物理-only
的目标就会悄悄把它们拖走，而 loss 曲线上完全看不出来（`5f1eff7` 修的就是这个）。

待决：

1. **关系怎么声明**：stage-2 的 config 里写一句 `continues_from: segvggt_finetune_agnostic`，
   由校验层去推导"必须冻的集合"？还是把关系写在 lock 文件里？
2. **约束是什么**：`frozen(stage2) ⊇ trained(stage1)`？还是更松的
   `trained(stage2) ∩ trained(stage1) = ∅`？两者对 joint 配方的含义不同——
   joint 同时训 `instance_` 和 `query_physgm`，它不是任何 stage 的续作。
3. **ckpt 侧要不要查**：`pretrained_weights` 加载的那个 ckpt 里，实际存在哪些键、
   哪些是随机 init（`query_physgm` by design 是 missing，`segvggt.py:297-320` 已有
   `_assert_query_physgm_coverage`）——这个已有的断言要不要并进指纹层。

**已有素材**：`src/model/arch/segvggt.py:297-320` 是这类断言的现成样板，可以直接抄形状。

## 完成判据

跨 stage 约束的形式定下来，且能对现有三份 segvggt 配方（stage-1 / joint / physgm）
各说清它该被约束成什么。

---

## 决议（2026-09-04）：整票划出 scope，不在本图落地

### 现场读数（三条，先于决策）

- **F-a · 今天全仓只有一条真正的 stage 链边。** 五份 IGGT 配方（`instseg_iggt` /
  `phys_iggt` / `phys_prop_iggt` / `physgm_iggt` / `physgm_dpt_iggt`）**全部**加载同一个
  外部基座 `iggt_checkpoint.pth`，彼此不是续作；`segvggt_scannet` 是 `""`；
  `segvggt_agnostic_phys_joint` 与 `segvggt_finetune_agnostic` 加载官方
  `segvggt_scannet200.pt`。唯一一条「A 的产物喂给 B」是 `segvggt_physgm.yaml:45`
  → `output/exp_segvggt_finetune_agnostic/2026-07-28_21-42-50/checkpoints/last.ckpt`。**n = 1。**

- **F-b · 待决第 2 问的两条候选是同一条约束。** 在「每个参数非冻即训」下
  `frozen(stage2) ⊇ trained(stage1)` ⟺ `trained(stage2) ∩ trained(stage1) = ∅`。
  唯一差别是 **stage-1 训过、但在 stage-2 的模型里根本不存在**的参数：`⊇` 报错、`∩=∅` 通过。
  所以真正待决的从来不是「选哪条」，而是「参数消失算不算断链」。

- **F-c · 前驱身份今天不是静态可知的。** 链边的唯一载体是 `pretrained_weights`——
  一个可被 CLI 覆盖的文件路径，且 `segvggt_physgm.yaml:4` 自陈可以是「stage-1 .ckpt
  **或**官方 .pt」。config 里写 `continues_from:` 是一句**无人核对的声明**。
  另一侧：`segvggt.py:358` / `iggt.py:333` 拿到的是**完整 ckpt dict**（`state_dict`
  只是其中一个键），所以往 ckpt 里盖指纹戳、由后继在加载时读出来，物理上做得到。

### D1 · 不建跨 stage 校验层（待决第 1、2 问一并作废）

三个走法各自的代价：

- **(a) 静态声明** `continues_from: <experiment>`，校验层读前驱 **lock** 求出
  `trained(stage1)` 判交。纯 CPU、与 09 的 lock 层共用读写。**但 F-c**：声明与实际加载的
  ckpt 无任何绑定，换个 `pretrained_weights` 路径校验照样通过。它守的是「配方之间的意图
  关系」，不是「这次 run 真的续了谁」。
- **(b) ckpt 盖戳**：训练结束把本次 lock 全文写进 ckpt 自定义键，后继加载时读真实前驱。
  唯一的真相源版本。**但**对官方 `.pt`（无戳）必须放行 ⇒ 22 份里 21 份走的正是这条无戳
  路径，护栏在绝大多数 run 上静默；且要动 checkpoint 写入路径，是 09 之外的第二处落地。
- **(c) 不建**。**选它。**

选 (c) 的实质理由：**本图对 stage 链断裂的实际防线是 lock 的 `git diff`。**
`5f1eff7`（camera_token 漏冻）在 lock 体制下就是 diff 里多出的两行，人在 review 时看见。
cross-stage 约束比 lock 多抓的只有一类：**配方第一次就写错**（错误的可训集合随 lock 初次
生成一起提交，diff 干净）。为 n = 1 条链边、且这条边实测完全自洽
（`trained(s1) \ frozen(s2)` = 0、`trained(s1) ∩ trained(s2)` = 0）建第二套机制，收益对不上。

**留给未来的接口（不是兼容层，是「不需要接口」的结论）**：03 定的指纹正文已经是**逐参数名**，
两份 lock 求交本就够用。真要建 (a) 时它是一个**读两份 lock 的独立脚本**，不回头改指纹格式
也不改 09 的 capture 函数。→ 已在 09 号票记一行。

**F4（`phys_iggt` 在物理阶段仍训 `point_head`/`depth_head`，与同路线邻居相反）随本票一起出图**：
F-a 已证明这五份不构成链，它们是同一基座的五个平行分支，「相反」不成立——各自的可训集合
由各自的 lock 锁住，够了。

### D2 · `_assert_query_physgm_coverage` 不并进指纹层（待决第 3 问）

它查的是「ckpt 里有 `query_physgm` 权重就必须全部加载成功」，属于**权重的值是否落位**。
冻结三判据是不建计算图 / 不进优化器 / 权重不变——它**一条都不占**。

把它并进来，和 02 号票清除的 `freeze_module` 是**同一类错误的两面**：那次是把不同机制混进
一个入口，这次是把不同判据混进一个层。

**处置**：留在 `src/model/arch/segvggt.py:297-320` 原地，一行不改；改为在 10 号票的
`CONTEXT.md` 术语表里立词把它与指纹**显式隔开**（「权重落位断言 ≠ 指纹」）。→ 已写进 10 号票。

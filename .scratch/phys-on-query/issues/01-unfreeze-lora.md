# 01 — 要不要放开 LoRA（冻结底座 vs LoRA joint）

Type: grilling
Status: closed
Assignee: krNeko9t
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

---

## 解决（2026-09-07）

**答案：不是三选一，是 (d) 两臂并行对照。** 4 卡跑 (a) 全冻底座、4 卡跑 (b) 放开 LoRA，
同时起，唯一变量是一条 `!*.lora.*`。

### 判决依据：票面的成本表被实测推翻了两条

**代价一（显存/时间）作废。** 从官方 ckpt 逐张量数出的真账：

| 桶 | 张量 | 参数量 | 现状 |
|---|---|---|---|
| `instance_` | 1011 | 454.25 M | 训 |
| `semantic_head` | 62 | 32.63 M | 训 |
| `query_physgm` | 15 | ~0.2 M | 训（随机 init） |
| block MLP | 192 | **402.90 M** | 冻 |
| block attn qkv/proj 底座 | 192 | 201.52 M | **构造期冻**（`layers/lora.py:123-125`） |
| block norm/ls | 480 | 0.31 M | 冻 |
| **block LoRA** | 192 | **9.44 M** | 冻 ← 本票的标的 |
| `patch_embed` | 344 | 304.37 M | 冻 |
| `camera_head`+`depth_head` | 131 | 248.83 M | 冻 |

放开 LoRA = 487.1M → 496.5M，**+1.9%**，AdamW 动量 +~75MB/卡。**显存代价约等于零**，
票面"显存与训练时间的账要重算"这条划掉。

**票面的警告则被数字坐实**：直接删 `frame_blocks`/`global_blocks` 多解冻 **+412.6M**
（几乎全是 MLP），可训参数近乎翻倍。但它**不等于全量微调**——attn qkv/proj 底座由 LoRA
在构造时冻死，`apply_freeze()` 只增不减（`base_wrapper.py:516-522`）碰不到它。
准确说法是"顺带解冻整个 block MLP"。

**时间不是稀缺资源。** 2026-07-28 那次 joint 跑实测 step 0→500 用时 **10 分 49 秒**
⇒ **1.30 s/step**，20k step ≈ **7.2 小时**（8 卡 / bs=1 / 4 视角 / 252×448）。
距 09-25 还有 19 天。**(c) 的"先 a 后 b"是在算力紧张的假设下才成立的排序，而该假设已被证伪。**
（那次跑到 step 500 就停了，`every_n_train_steps: 1000` ⇒ **没存下任何 ckpt**；
当时 val `ap50=0.2602 / matched_iou=0.4788`。）

**代价二（几何漂移）现状下不可观测。** `camera_head`/`depth_head` 全冻 + `segvggt_geo.weight: 0`
⇒ 没有任何 loss 在盯几何，也没有任何指标在报它。所以 LoRA 一开，几何漂没漂**根本看不见**，
不是"漂了会被罚"。判决：**几何不作为本图交付**（终点只写 class + P），但 (b) 臂加
**只读的 depth/pose 漂移指标**（不进 loss，`segvggt_geo.weight` 保持 0）——
它是对照表的第二列。⇒ [11 号票](11-geo-drift-readonly-metric.md)

### 怎么放开：裸子串在表达力上写不出来

票面写的"从 `freeze_keywords` 去掉 `frame_blocks`/`global_blocks` 并补上块内 MLP/norm 的冻结"
**实测不可行**。`freeze_keywords` 是裸子串 OR 匹配（`base_wrapper.py:515`），
补冻用的关键词 `.mlp. / .norm1. / .norm2. / .ls1. / .ls2. / .attn.` 会连带冻死
`instance_cross_blocks`(302.32M) + `instance_query_self_attn`(151.30M)——**把要训的东西冻了**。
子串 OR 做不出"除了 lora 之外"。逐块枚举（24×2 块 × 5 组件 ≈ 240 条关键词）能work但是净负债。

⇒ 结论：**匹配语言必须能表达"除了"**。⇒ [09 号票](09-freeze-matching-language.md)（改法 A：
fnmatch glob + 前缀 `!` 取反，同一循环 last-match-wins）。(b) 臂的 config 等它，
**(a) 臂 config 一字不改、不等它**。

### 写进 config 的两个数

- **`!` 只放 LoRA**：`["*patch_embed*", "*frame_blocks*", "*global_blocks*", "*camera_head*",
  "*depth_head*", "*camera_token*", "*register_token*", "!*.lora.*"]`。
  block 的 MLP / norm / ls / attn 底座全留冻。
  **norm+ls(0.31M) 不一起放**——它是 BitFit 那类"参数极少效果不成比例"的位置，值得试，
  但那是**第三臂**，混进来这次对照就不是单变量了。
- **LoRA 的 lr = 默认组 1e-4**，不加 `param_groups`。理由：LoRA 不是随机 init，是官方 ckpt 里
  已在 ScanNet200 上训好的适配器，性质同 `instance_`/`semantic_head`；且两臂共享同一套 lr 设置
  ——又一次保住单变量。论文 A.4 的 pre-existing 6e-5 是在"冻结集只有 `patch_embed`、可训 1148M"
  的配方下调的，与这里可训 496M 不可比。

### 顺带落地的两条事实

- **lock 层已经在了**（`freeze-contract` 图 2026-09-04 到达终点，`b53f7dd`）。
  `config/experiment/locks/*.lock` 22 份全覆盖、缺 lock 启动硬错、正文逐行比对。
  ⇒ 本票任何 `freeze_keywords` 改动都必须重生成 lock，**`git diff` 就是每个 requires_grad
  翻转的参数各一行**。joint 那份现读数：**1533 行冻 / 1088 行训**，与训练日志
  "frozen 1533 params total" + `configure_optimizers` 15+1073 完全对上。
- **`optimizer.lr` 的注释是过期的**：实际默认组 1e-4、`query_physgm` 5e-4，
  注释（`:44-48`、`:65`）描述的是 2e-5 / 1e-4。这是 freeze-contract 01 号票的 F1，
  被那张图明文判出 scope。对两臂等同施加、不影响本次对照，但直接影响 checkpoint 质量。
  **不在收本票时顺手改**（顺手改 = 一次没人复核的配方变更）⇒ [10 号票](10-joint-lr.md)。

### 对下游

- → [05 号票](05-training-operating-point.md) 解除阻塞（01、02 均已关），且它现在要给
  **两臂**定运行点（卡怎么分、数据用 scannet100 还是 Infinigen 全量、步数）。
  (b) 臂的 config 另需 09 落地。
- → 新增 [09](09-freeze-matching-language.md) / [10](10-joint-lr.md) / [11](11-geo-drift-readonly-metric.md)。
- → 本图 Out of scope"冻结机制本身"那条**开了一个窄口**（见 map）。

---

## 实测结论（2026-09-08，两臂各跑满 20000 step）

**答案分岔：放开 LoRA 在分割上明确赢，在物性上三项都略输。而本图的终点是物性。**

两臂 09-07 17:09 同起，均正常收尾，各 40 个 val 点。A 用时 6.8 h（1.22 s/step），
B 9.0 h（1.62 s/step）。交付（根 `/mnt/storage_pool/liaoyuanjun/runs/`）：
`exp_phys_query_arm_{a_frozen,b_lora}/2026-09-07_17-09-03/checkpoints/epoch_114-step_20000.ckpt`
（10.5 / 10.6 G）。

**对照是干净的**：step 0 的 val 读数两臂逐字节相同；`loss/segvggt_phys_num` 逐点相同
（同批场景同顺序）；`*_const` / `*_clut` / `phys_n_matched` 在同一 step 上两臂相同——因为
`_optimal_assignment`（`src/evaluation/instance_metrics.py:139`）在 `[Q=400, K≈30]` 上把
K 个 GT **全部**配上，打分集合只由 GT 决定。⇒ **同 step 跨臂是唯一不受采样污染的读法。**

### 分割：40 点 × 3 指标 = 120/120

| | B/A 均值 | 最小 | 末点 A → B |
|---|---|---|---|
| `val/ap50` | **+12.7%** | +4.0% | 0.5927 → 0.6586 |
| `val/ap` | +9.9% | +3.6% | 0.3789 → 0.4197 |
| `val/matched_iou_mean` | +3.3% | +1.3% | 0.6516 → 0.6700 |

### 物性：抛硬币

全程 40 点均值（学生/clut，<1 为赢过查表）与 B 的胜场：

| | A 冻结 | B LoRA | B 胜 |
|---|---|---|---|
| log10 E | **0.959** | 0.971 | 19/40 |
| log10 ρ | 1.005 | **1.004** | 26/40 |
| ν | **1.019** | 1.030 | 14/40 |

末 5 点主表口径（对常数基线）：

| | A | B | 类别查表 |
|---|---|---|---|
| log10 E | **0.871** | 0.903 | 0.901 |
| log10 ρ | **0.959** | 0.971 | 0.961 |
| ν | **0.967** | 0.989 | 0.939 |

**A 臂最终比常数好 12.9/4.1/3.3%；与查表比 E 赢 3.3%、ρ 打平、ν 输 3.0%——三项只有一项赢。**

### 时间形状：物性 5–10k 触底后回退，分割到 20k 未见顶

A 臂按 5000 step 分段（学生/clut）：E 0.983 / **0.936** / 0.950 / 0.969；
ρ 1.022 / 0.992 / 0.999 / 1.005；ν 1.019 / 1.001 / 1.028 / 1.027。
同期 `ap50` 0.456 / 0.540 / 0.576 / **0.593**，`iou` 0.581 / 0.627 / 0.643 / **0.651**。
一斜一平，且 B 分割大幅领先却换不来物性 ⇒ **二者脱钩。**

### 对下游

1. **交付哪个 ckpt 取决于主表报什么**——按分割选 (b)，按物性选 (a)。本票不替下游定。
2. **瓶颈不在底座冻不冻**：9.44M LoRA 抬了 12.7% ap50 却动不了物性一分 ⇒ 限制在**头或监督
   信号**。该不该改头，先看 [14 号票](14-checkpoint-diagnosis.md)，别直接开训。
3. ⚠️ 本节数字受四个度量缺陷影响 ⇒ [13 号票](13-measurement-trust.md)。受影响最大的是跨
   step 趋势（基本不可读），最小的是同 step 跨臂——**上面两条主结论恰好都建立在后者上。**

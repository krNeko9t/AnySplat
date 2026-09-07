# Map: 让 SegVGGT 的 query 额外输出 object-centric P

Label: wayfinder:map

## Destination

跑出**第一个可交付的 checkpoint**：SegVGGT backbone + pretrained weight，query 在预测
class 之外额外输出 object-centric 的 P = (杨氏模量, 泊松比, 密度)，一次前馈、场景级、
推理不需要 GT mask。

**判据不是"物性预测得准"，是"学生逼近教师"**——即逼近当前这条 VLM 伪标签管线的上限。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）：导师要的是 checkpoint，
不是方案。票可以是动手的。

**领域**：前馈多视角 3D 场景理解 + 物性预测。ICLR 2027 截止 **2026-09-25**，
目标是**按时投出，不求录用**（memory `paper-goal-iclr2027`）。

**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

**机器**（2026-09-07 复核）：本图的活跑在 **GPU 服务器 `bms-39468022-001`**（8×A100-40G，
核实时 0 MiB 占用），conda env **`anysplat`**（base 里没有 torch）。`硬件环境.md` 说的
"开发机无 GPU"指的是另一台；该文件现已随仓库在本机。
数据全在本地盘 `/mnt/storage_pool/liaoyuanjun/data/InsScene-15K/`，**没有"传数据"这道工序**。
`research_space/`（事实底座）2026-09-07 已从开发机拷到本机，但**不在 git 里**——
换机器要重新拷。

### 本图开工前已定的事（2026-09-03 grilling，不再重开）

1. **上限 = 教师**。前馈视觉底座在这个任务上的天花板就是生成伪标签的那个 VLM 管线。
   照片里没有杨氏模量；任何前馈模型在极限上都是查表，只是表的分辨率不同。
   **这条要写进论文第一段，自己说出来，不让审稿人来问。**
2. **上限由（输入数据分布 × VLM 本身 × 标签生成方法）三者决定**，未来一定会重跑标签来抬高它。
   **抬高上限 = out of scope；逼近上限 = 本图全部内容。**
3. **论文主张 = 系统 + 度量**：贡献是"把一条昂贵的 per-instance VLM 管线摊销成一次前馈，
   且 P 落在正确的实例上"，不是"模型理解物理"。同时诚实量出其中有多少是类别查表
   （已发表工作的主表里没人报过这个 trivial baseline）。
   "打不过查表"因此不是弱点，是自己报出的发现。
4. **物性头保持最朴素**：query → MLP → (mu, log var)，Gaussian NLL，不进 Hungarian 匹配代价。
   花招（如"类别项 + 残差项"分解）一律押后，等第一个 checkpoint 出来看它烂在哪再说。
5. **不以现有代码实现为准**。现有半成品只作为事实来源（`research_space/opus_2/notes/code_facts.md`
   带 file:line），不作为设计依据。

### 事实底座（可复核，别重新推导）

- `research_space/opus/01_审视结论.md` — 逐节点审视 + 可进论文的数字
- `research_space/opus_2/01_四问答复.md` — 四问答复 + 对上一轮的修正清单
- `research_space/opus_2/notes/code_facts.md` — 现场读码事实，带 file:line

关键数字：材质熵 H=3.865 bit，H(P|类别)=1.444 bit ⇒ **类别解释 62.6%，残差 37.4%**（Infinigen shader，
非 VLM 标签）；同场景同类别 ≥2 实例的实例占比 99.2%；29.6% 的物体跨材质原型。
**逐张量实测参数量**（2026-09-07，从官方 `segvggt_scannet200.pt` 直接数，01 号票）：
`instance_` 454.25M / `semantic_head` 32.63M（= 现可训 487M）、**block MLP 402.90M**、
block attn qkv/proj 底座 201.52M（LoRA 构造期冻）、block norm/ls 0.31M、**block LoRA 9.44M**、
`patch_embed` 304.37M、geo 头 248.83M。
**实测吞吐**：1.30 s/step（8×A100 / bs=1 / 4 视角 / 252×448）⇒ 20k step ≈ 7.2 小时。
`segvggt_agnostic_phys_joint.lock`：1533 行冻 / 1088 行训。

SegVGGT Table 7：冻结 23.4 → LoRA joint **31.9**；Table 8：冻结底座下加大 head 22.4/23.4/**16.7**（倒退）。
⚠️ 以上是**论文数字，非本仓库实测**，不同数据/配方下不可直接套用。

## Decisions so far

<!-- 一行一个已关闭的票 -->

- [01 — 要不要放开 LoRA（冻结底座 vs LoRA joint）](issues/01-unfreeze-lora.md)：
  **不是三选一，是 (d) 两臂并行对照**——4 卡全冻底座、4 卡放开 LoRA，同时起，
  唯一变量 `!*.lora.*`（+9.44M / **+1.9%**，AdamW 动量 +75MB ⇒ **显存代价约等于零**，
  票面"代价一"作废）。改用实测而非论证，因为**时间不是稀缺资源**：实测 **1.30 s/step**
  ⇒ 20k step ≈ **7.2 小时**，19 天里能跑 60 次，(c) 的"先 a 后 b"所依赖的算力紧张假设被证伪。
  **代价二（几何漂移）在现状下不可观测**（geo 头全冻 + `weight: 0` + 无指标）⇒ 判几何
  不作为交付，代之以只读漂移指标（→ 11）。**票面给的改法实测不可行**：裸子串 OR 表达不出
  "除了 lora"，补冻关键词会连带冻死 `instance_cross_blocks`(302M)+`instance_query_self_attn`(151M)
  ⇒ 匹配语言要加"除了"（→ 09，**本图 Out of scope 因此开了一个窄口**）。
  norm+ls(0.31M) 不一起放、LoRA 用默认组 1e-4——两条都为保住单变量。
  顺带查出 `optimizer.lr` 注释过期（实际 1e-4/5e-4，注释写 2e-5/1e-4）⇒ → 10。

- [02 — 把 Infinigen 全量 VLM 伪标签接进训练](issues/02-labels-into-training.md)：
  标签本就在训练机本地（1466 场景 / 146,034 帧 / 53,328 条），缺的只是 manifest。
  已补 4 个脚本 + 3 份数据侧产物，`DatasetManifest + instascene_vlm_physgm` 已实测吐出
  `(类名, E, ν, ρ)`。id 空间三方统一（seg 像素值 = `Objects.object_index` = 标签 `id`），
  类别可从 `Objects_*.json` 97.5% 无损恢复。**发现在用的 z-score 常数是抄 PhysGM 的、
  与本语料严重不符（→ 07）**；`room:*`+`Window` 占 28.6% 标签（→ 08）。

## Not yet specified

- **物性头的花招**：`P̂ = LUT[ĉ] + Δ(q)` 这类"类别项 + 残差项"显式分解。等第一个 checkpoint
  的失败模式出来再判断值不值。已知代价：依赖闭集 200 类的 `ĉ`，认错类时误差不再平滑。
- **学生超越教师的那条合法路径**：教师只看一个最佳视角，学生看全部视角 ⇒ 学生可以更一致、
  更抗噪。这是唯一不违反"上限=教师"的超越方式，可测，但度量怎么定还没想清楚。
- **part-level 粒度**：29.6% 的物体跨材质原型，object-level 单标签对它们是系统性错误。
  query 范式下拆 part 最便宜（多分配几个 query + 把 GT 拆到 part 粒度，不改表示）。
- **训练/推理的 train-test 失配复核**：query 路径按 `code_facts` D 应当自动免除 GT-mask 依赖，
  但要在真实运行里确认一遍。
- **标签里被丢掉的那些字段**：每条伪标签都白送 `object_description` / `appearance_materials`
  (prototype+score) / `physical_priors` (bin+confidence) / `n_views` / 每个量的 `variance`，
  目前一个都没进 loss。哪些值得用、怎么用（样本加权？辅助监督？）还看不清，等 04 判完教师余量的
  性质再说。

## Out of scope

- **抬高上限**：重跑伪标签（换视角选择 / 改 prompt / steering / agentic harness / 接外部材料数据库）。
  一定会做，但不在这张图里——这张图只做逼近当前上限。
- **视频监督反演物性**（看物体形变优化参数）。原理上是唯一真有视觉物理信号的路，我们做不到。
- **冻结机制本身**（字符串匹配脆弱、漏冻 token、多种实现并存）：
  另开一张图 `.scratch/freeze-contract/map.md`（**该图已于 2026-09-04 到达终点**，10 票全关）。
  本图只消费冻结配方，不改机制。
  **2026-09-07 开了一个窄口**（由 [01 号票](issues/01-unfreeze-lora.md) 逼出）：
  freeze-contract 把"设计一套新的冻结 API"判出 scope 的理由是「换任何声明式语法照样要枚举，
  会咬人的是枚举漏了没人发现，那是校验问题」。01 号票实测**证伪了这条前提的一半**——
  问题不是"照样要枚举"，是**裸子串在表达力上写不出来**："block 里除了 lora 全冻"这条合法配方
  根本不存在对应写法。**仅此一个缺口**收进本图为 [09 号票](issues/09-freeze-matching-language.md)
  （glob + `!` 取反，一个循环、零新增分支、两道护栏原样存活）。
  **不重开 freeze-contract**——它的终点（验收层）确实建成了，且已在该图 Out of scope
  留了指回本票的一行。**这个口只开这么大**：除"表达'除了'"之外的任何机制改动仍在本图之外。
  **2026-09-04 更正**：原文把"bf16 静默冻结"也算进这条，是错的——它 `requires_grad` 全程为真、
  参数进了优化器，只占冻结三判据的第三条，**不是冻结**。freeze-contract 图已据此把它
  整票判出 scope（该图 07 号票），本图不能再把它推回去，否则两张图互指、它谁都不归。
  现寄存为本图 [06 号票](issues/06-aggregator-precision.md)（明确标注不在路上、不阻塞任何票）。
- **IGGT 路线**（`phys_iggt.yaml`）：aggregator/part_adaptor/part_head 全冻，phys head 是纯 frozen
  probe；要修得先给 VGGT aggregator 加 adapter，成本高一个量级。降级为"冻结底座"那一行的对照组。
- **复现 PIXIE / VoMP / PhysGS，MPM 灵敏度实验，HILO / PixieVerse**（memory `paper-direction-scene-phys`
  明令不碰）。

# Map: 让 SegVGGT 的 query 额外输出 object-centric P

Label: wayfinder:map

## Destination

跑出**第一个可交付的 checkpoint**：SegVGGT backbone + pretrained weight，query 在预测
**实例分割**之外额外输出 object-centric 的 P = (杨氏模量, 泊松比, 密度)，一次前馈、场景级、
推理不需要 GT mask。

（2026-09-07 [05 号票](issues/05-training-operating-point.md)更正措辞：原文写的是"预测 class
之外"，但 joint 配方是 `class_agnostic: true`，`loss_segvggt.py:274-277` 把 200+1 的分类头
logsumexp 成 2 列 ⇒ **这条路上的 checkpoint 不预测类别，只预测 objectness**。
这是有意的 Phase-1 配方，不是漂移；**类别预测因此明确出 scope**。）

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

⚠️ **训练输出必须写 `/mnt/storage_pool`，不能写仓库下的 `output/`**（2026-09-07
[05 号票](issues/05-training-operating-point.md)现场吃到）：根盘 `/dev/sda1` 只有 439G 且长期
**93% 满**，而本配方一份 checkpoint 就是 **9.8 G**（`save_weights_only: false` ⇒ 1.37B fp32
权重 + 487M 可训参数的 AdamW 状态）；Lightning 还会在 `save_top_k` 之外另留 `last.ckpt`
⇒ **每臂常驻约 20 G、写入峰值近 30 G**。两臂一起跑直接把根盘写满，
arm (a) 在 step 1990 死于 `OSError: [Errno 28] No space left on device`。
两份臂 config 的 `hydra.run.dir` 已改到 `/mnt/storage_pool/liaoyuanjun/runs/`（3.5T，766G 可用）。
`research_space/`（事实底座）2026-09-07 已从开发机拷到本机，但**不在 git 里**——
换机器要重新拷。

⚠️ **跑任何 python 都要带 `PYTHONNOUSERSITE=1`**（2026-09-07 [11 号票](issues/11-geo-drift-readonly-metric.md)
现场发现）：`~/.local/lib/python3.10/site-packages` 里有个 2026-09-05 装的 **torch 2.7.1+cu118**，
user-site 优先级高于 env，把 `anysplat` 自己的 **torch 2.4.1+cu124** 顶掉，
`pytorch3d._C` / `torch_scatter` 立刻 undefined symbol，**训练根本起不来**。
加上这个变量即好，env 本身没坏。

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
**实测吞吐**：1.30 s/step（8×A100 / 4 视角 / 252×448）⇒ 20k step ≈ 7.2 小时。
⚠️ 那次是 **scannet100 且开着抖动**跑的。05 号票钉死视角/分辨率并换到 Infinigen 后**重测为
≈1.1 s/step**（4×A100 / 4 视角 / 252×448 / 每卡 2 场景）⇒ 20k step ≈ **6.1 小时/臂**。
**`data_loader.train.batch_size` 在这条路径上是死的**（`data_sampler.py:118`）：
每卡场景数 = `floor(max_img_per_gpu / 视角数)` = `floor(8/4)` = **2**，4 卡 ⇒ 8 场景/step。
`segvggt_agnostic_phys_joint.lock`：1533 行冻 / 1088 行训（05 号票的 (a) 臂 lock 与它逐字节相同）。
**训练划分**（05 号票）：`config/experiment/splits/infinigen_phys_split.json`，
1461 个可用场景 ⇒ **1387 训 / 74 验**，按 `scene_XXX` 生成分片整组切（实测分片间不共享资产：
物体名 Jaccard 组内 0.049 vs 组间 0.047）。
**`PHYSGM_NORMALIZATION` 已换成本语料训练集拟合值**（05 号票执行、07 号票 2026-09-07 追认关票）：
density 2.863740/0.399147、E 9.495947/1.317972、ν 0.336525/0.066235。
⇒ **所有用 `instascene_vlm_physgm` 的旧 ckpt 反归一化都对不上了**（实际只有一份：
`output/exp_segvggt_physgm/2026-07-29_07-35-57__STALE-PHYSGM-NORM`，已改名标记，07 号票）。
**留出集上的类别查表基线**（05 号票，2724 实例）：log10 E 常数 **0.9050** → 查表 **0.7477**（−17.4%）；
ρ 0.2708→0.2260；ν 0.0504→0.0424。分片间波动不小（另一份划分给 −12.5%），别当三位有效数字。
**VLM 伪标签的类内方差分解**（2026-09-07，全量 51,981 条，04 号票，与 parser 同量纲）：
log10 E / log10 ρ / ν 的类内占比 **73.6% / 76.7% / 78.2%**（截尾后 72.7%，非离群值所致）；
**描述-类名不一致率 49.3%**，一致子集类内 **46.4%**、不一致子集 **87.5%**；
`prototype` 解释类内残差 60.4%（一致子集）；类别查表 vs 常数基线只降 **15.6%**（log10 E MAE 0.890→0.751）；
213 类中 **126 个单例类** ⇒ 只能按场景划 train/val。
脚本 `scripts/analyze_teacher_headroom.py`，全部数字在 `$R/teacher_headroom_infinigen.json`。

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
  **2026-09-08 实测收口（两臂各跑满 20000 step）**：答案分岔——**分割 40 点 × 3 指标
  120/120 全胜（ap50 +12.7%），物性三项全输且是抛硬币（胜场 19/26/14）**。
  A 臂最终比常数基线好 12.9/4.1/3.3%，与类别查表比 **E 赢 3.3%、ρ 打平、ν 输 3.0%**。
  物性 5–10k 触底后回退，分割到 20k 未见顶 ⇒ **二者脱钩，瓶颈在头或监督信号不在特征。**
  ⇒ 交付哪个 ckpt 取决于主表报什么；改头之前先做 [14 号票](issues/14-checkpoint-diagnosis.md)
  的诊断。⚠️ 全部数字受四个度量缺陷影响 ⇒ [13 号票](issues/13-measurement-trust.md)，
  其中"val 视角未固定 seed"和"中间 ckpt 被 `save_top_k:1` 删光"**必须在下一次起跑前解决**。

- [02 — 把 Infinigen 全量 VLM 伪标签接进训练](issues/02-labels-into-training.md)：
  标签本就在训练机本地（1466 场景 / 146,034 帧 / 53,328 条），缺的只是 manifest。
  已补 4 个脚本 + 3 份数据侧产物，`DatasetManifest + instascene_vlm_physgm` 已实测吐出
  `(类名, E, ν, ρ)`。id 空间三方统一（seg 像素值 = `Objects.object_index` = 标签 `id`），
  类别可从 `Objects_*.json` 97.5% 无损恢复。**发现在用的 z-score 常数是抄 PhysGM 的、
  与本语料严重不符（→ 07）**；`room:*`+`Window` 占 28.6% 标签（→ 08）。

- [04 — 教师自己留了多少余量（同类别内标签方差）](issues/04-teacher-headroom.md)：
  **教师留了约 74% 的类内余量，但其中约一半是 grounding 噪声。** 全量 51,981 条实测
  类内/全体方差占比 log10 E **73.6%** / log10 ρ 76.7% / ν 78.2%（截尾后 72.7%，
  **不是离群值撑的**）⇒ `H(P_VLM|类别) ≈ 0` 正式证伪，**最坏情形排除，学生有东西可学**。
  但这个数是客观 shader 残差（37.4%）的**两倍**，多出来的那倍被抓到了：
  **49.3% 的标签，VLM 自己写的 `object_description` 与该 id 的真实类名对不上**
  （`CeilingLight`→冰箱、`Window`→会议桌）；一致子集类内 **46.4%**、不一致子集 **87.5%**,
  而 46.4% 正好回到客观残差的量级。**排除了"类别表错位"**（id 偏移 k=±1 未富集），
  认错目标弱富集于同场景内（94.9% vs 机会基线 85.1%），指向"给 VLM 看的图对错了物体"，
  **但机制未定案且属抬高上限 ⇒ 仍在 Out of scope**；"训练要不要区别对待"属逼近上限
  ⇒ → [12](issues/12-label-reliability.md)。
  **余量是结构化的**：`appearance_materials.prototype` 解释掉类内残差的 60.4%（一致子集内）
  ⇒ 教师在做逐实例材质判断，不是给随机数。
  **免费捎带**：(a) 类别查表只比常数基线好 15.6%（log10 E 0.890→0.751）⇒ trivial baseline 很弱，
  对本图是好消息；(b) **按类别划 train/val 不可行**（213 类里 126 个单例），只能按场景划 ⇒ 直接答了
  [03](issues/03-approach-ceiling-metric.md) 第 4 点；(c) 非物体类 log10 E 类内 **95.6%** vs 真物体 65.9%
  且 grounding 一致率相同 ⇒ 直接答了 [08](issues/08-non-object-classes.md) 第 4 点，
  但**最脏的其实是软体真物体**（Pillow/Blanket/Towel sd≈2.0）⇒ 归 12 不归 08。
  局限：一致/不一致是非随机划分，46.4% 是**带选择偏差的下界**；全程没看过一张图。

- [11 — depth/pose 漂移的只读指标](issues/11-geo-drift-readonly-metric.md)：
  **做完了**。`LossSegVGGTGeo.metrics()`（`@no_grad`，复用 `_camera_loss`/`_depth_loss`，
  不是新写一套）+ wrapper `_log_geo_drift()`，validation 里报 6 条 `val/geo_*`：
  `geo_camera` 与末次迭代的 **T/R/fl 分解**、`geo_depth` 与无量纲的 **`geo_depth_rel`**。
  指标在 wrapper 里不在 config 里 ⇒ **两臂自动都有**。
  实测 `camera=0.0079 (T=0.0134 R=0.0001 fl=0.0022) depth=106.3 depth_rel=0.052`；
  `weight: 0` 不动、`Trainable params 487M` 不变、**lock `unchanged`**。
  **加 `geo_depth_rel` 是实测逼出来的**：GT depth 是**毫米**（median 2263）⇒ 106 其实是
  4.7% 相对误差，绝对值跨臂不可比。顺带记下：训练侧 depth 项 117.6 vs camera 0.022 而
  `lambda_camera:5 / lambda_depth:1` ⇒ **谁哪天打开 `segvggt_geo.weight`，depth 会以约 1000× 压死 camera**
  （现被 `weight:0` 挡着，本图不开 geo loss，不处理，只留话）。

- [05 — 三周内能跑完的运行点](issues/05-training-operating-point.md)：
  **两臂运行点定死，(a) 臂已在 4 卡上跑起来。** 数据 = **Infinigen 全量**
  （1387 训 / 74 验，按生成分片整组切；选它的决定性理由不是数据量而是**参照系**——
  04 号票的全部刻度只在这份语料上存在，scannet100 上的 checkpoint 主表没有横坐标）；
  **无 curriculum**（预训练已完成"几何→实例"两段，且仓库根本没有该机制）；
  **Q=400 锁死**（`instance_query_token` 在权重里就是 `(1,400,1024)`，改了就随机初始化整个 query bank，
  票面第 3 项自动答完）；20k step；lr 维持 `1e-4/5e-4`（**10 号票改成从两臂曲线读答案**——
  降到 2e-5 会让 (b) 臂的 LoRA 几乎不动，把对照读成噪声）。
  **查出四件"不改就读不出东西"的事，都就地改了**：(i) train/val 此前共用同一份 dataset cfg
  ⇒ **val 跑在训练场景上**；(ii) `DynamicBatchSampler` 有两个没人提过的抖动（视角数 ∈ {2,3,4}、
  输入高 ∈ [252,448]，H=448 端**裁掉 44% 画幅**）⇒ 钉死；(iii) z-score 常数是抄来的
  ⇒ **07 号票的阻塞随划分落地当场消失**，就地换成训练集拟合值；(iv) **val 里一个物性数都没有**
  ⇒ 新增 `src/evaluation/physics_metrics.py`，学生/常数/类别查表三行同批对照。
  留出集实测 log10 E 常数 **0.9050** → 类别查表 **0.7477**（−17.4%），与 04 的 −15.6% 一致。
  顺带修了个潜伏 bug：`DatasetManifest` 的"帧数够不够"过滤器比错了量（要 13 帧却比 4 帧），
  钉死视角把它从偶发崩变成必现崩；另外把训练输出挪出根盘（ckpt 9.8G，两臂能把 439G 的 `/` 写满）。
  **2026-09-07 17:08：09 号票落地后两臂已同时起跑**——`frozen 1533 vs 1341`（差 192）、
  `Trainable params 487M vs 496M`（差 9.44M / +1.9%，与 01 号票对上）、lock 恰好 192 行 `-`→`T` 全含 `.lora.`、
  **两臂 step 0 的 val 读数逐字节相同** ⇒ 对照的起点是同一个点，唯一变量就是 LoRA。约 6 小时后两条曲线到手。

- [09 — 冻结匹配语言加"除了"（glob + `!` 取反）](issues/09-freeze-matching-language.md)：
  **做完了，(b) 臂解除阻塞。** `apply_freeze()` 换成 fnmatch glob + 前缀 `!` 取反、后命中者胜，
  **仍是一个函数一个循环、零新增业务分支**；`!` 的语义是"该关键词不选中它"，函数里**没有
  `requires_grad = True` 这个赋值**，所以"只冻不解冻"和"零命中 raise"两道护栏原样存活
  （实测：只写 `["!*.lora.*"]` 时 192 张 LoRA 基座依旧是冻的；`!` 拼错照样炸）。
  固化成 `tests/test_freeze_matching.py`（6 个测试，毫秒级，不建底座）。
  **17 份配方机械迁移** `x`→`"*x*"`，**26 份 lock 重生成、每份恰好差 1 行**，
  且那一行必是 lock 头部回显的关键词——`structure` 哈希与逐张量正文一行未动。
  **(b) 臂 lock 恰好 192 行 `-`→`T`，全部含 `.lora.`，合计 9,437,184 参数 = 9.44M**，
  与 01 号票数出来的对上；`freeze_contract.verify()` 对两臂 + joint 三份配方实跑全 PASS。
  两处票面数字就地更正：**"22 份配方"实为 26 份 yaml / 18 份自己声明**；
  **判据"lock diff 必须全空"字面不可能成立**——lock 头部按设计逐字回显配方关键词，
  实质判据（可训集合零变化）成立且更强。
  ⚠️ **迁移唯一会咬人的地方**：裸写 `patch_embed` 现在匹配不到任何东西，会撞零命中硬错
  ——咬得很响，不会静默。`param_groups[].keywords` 仍是裸子串，没铺过去。

- [07 — z-score 常数用抄来的还是本语料拟合的](issues/07-physgm-norm-constants.md)：
  **四条全部追认，票关。** 维持训练集拟合值（density 2.863740/0.399147、E 9.495947/1.317972、
  nu 0.336525/0.066235）、维持硬编码 tuple（不走 config 指 JSON，本图只有一份语料，
  灵活性买不到东西却买来一个静默错误面）、**泊松比不换 logit**。
  **新查出一条比票面更硬的换常数理由**：`physics_metrics.py:125` 把 MAE 乘回 `std` 报原量纲
  ⇒ 度量本身对常数选择不变，但 :127 的**常数基线是 `z=0` = 训练集均值**——旧常数下 `z=0`
  是 *PhysGM 语料的*均值 ⇒ **会把论文主表的 trivial baseline 做成人为变弱的假基线**，
  而这正是 04 号票整张票的价值所在。
  **logit 的前提被实测证伪**：nu 的 `≥0.5` 只占 **0.014%**、`≤0` 占 0.036%（训练划分内 400 场景
  / 13,975 条），没有"边界堆积"要摊开；"z 放大 1.7 倍"是相对旧的**错** std 而言，
  per-property 标准化本来就该这样。⚠️ 但撞见一个票面没料到的形状问题：
  **nu 是量化网格不是连续量**（0.35 占 39.7%、0.3 26.6%、0.45 11.5%，前 5 值吃掉 87%）
  ⇒ Gaussian NLL 对它是错模型，属改头，按前提第 4 条押后（→ Not yet specified）。
  **旧 ckpt 护栏取最轻一档**：现场核实受影响的 ckpt **只有一份**（`exp_segvggt_physgm/2026-07-29_07-35-57`，
  另三份 ckpt 根本没用 physgm parser），已改名 `__STALE-PHYSGM-NORM` + 放 `STALE_NORMALIZATION.md`。
  诚实记账：**目录名只挡人不挡程序**，ckpt 拷走标记就没了；没做"常数哈希写进 ckpt、
  不匹配硬报错"那一档。

## Not yet specified

- **物性头的花招**：`P̂ = LUT[ĉ] + Δ(q)` 这类"类别项 + 残差项"显式分解。等第一个 checkpoint
  的失败模式出来再判断值不值。已知代价：依赖闭集 200 类的 `ĉ`，认错类时误差不再平滑。
  **2026-09-07 加料**（[04](issues/04-teacher-headroom.md)）：标签白送的 `appearance_materials.prototype`
  解释掉类内残差的 60.4%（一致子集内、log10 E）——比类名本身有用得多。
  把它当辅助监督头（query → 材质原型 → P）诱惑很大，但仍是**改头**，按 Notes 第 4 条押后。
  [12 号票](issues/12-label-reliability.md)已明确把它排除在外，只谈样本可靠性。
  **2026-09-08：第一个 checkpoint 的失败模式已经出来了**（01 号票：LoRA 抬分割 12.7% 却
  动不了物性一分）⇒ "等失败模式"这个条件已满足，但**先诊断再决定**——
  ⇒ [14 号票](issues/14-checkpoint-diagnosis.md)（不训练，在已交付 ckpt 上量学生是不是
  `LUT[类别]` 的复读；三种结果分别指向加头 / 改损失 / 改指标）。
- **学生超越教师的那条合法路径**：教师只看一个最佳视角，学生看全部视角 ⇒ 学生可以更一致、
  更抗噪。这是唯一不违反"上限=教师"的超越方式，可测，但度量怎么定还没想清楚。
  **2026-09-07 变具体了一点**（[04](issues/04-teacher-headroom.md)）：`n_views=1` 的 3,716 条
  类内方差占比 81.9%，显著高于 2/3/4 视角的 68–78% ⇒ **"视角越多标签越稳"在标签侧已有实证**。
  但它只占 7.2%，且 49.3% 的 grounding 错误在 4 视角上照样发生 ⇒ 多视角**没能**修好 grounding。
  这条路径要成立，得先想清楚学生凭什么修好教师修不好的东西。仍不够格开票。
- **part-level 粒度**：29.6% 的物体跨材质原型，object-level 单标签对它们是系统性错误。
  query 范式下拆 part 最便宜（多分配几个 query + 把 GT 拆到 part 粒度，不改表示）。
- **训练/推理的 train-test 失配复核**：query 路径按 `code_facts` D 应当自动免除 GT-mask 依赖，
  但要在真实运行里确认一遍。
  **2026-09-07 缩小了**（[05](issues/05-training-operating-point.md)）：`fixed_views_and_shape`
  已把训练分布钉成与 val / 推理相同（4 视角 / 252×448），所以"视角数与分辨率的失配"这一半
  已消。剩下的只有 GT-mask 依赖那一半。

- **泊松比可能整个 NLL 都是错模型**（2026-09-07 [07 号票](issues/07-physgm-norm-constants.md)实测撞见）：
  nu 不是连续量，是 VLM 吐出来的**量化网格**——`0.35` 占 39.7%、`0.3` 26.6%、`0.45` 11.5%、
  `0.2` 5.5%、`0.4` 3.9%，**前 5 个值吃掉 87%**。拿 Gaussian NLL 拟合近似分类变量，
  σ 学到的东西没有明确含义。是不是该换成对这 5 个值的分类头（或干脆不监督 nu）还没想清楚，
  且属**改头** ⇒ 按 Notes 前提第 4 条押后，等第一个 checkpoint 看它烂在哪。
  顺带：留出集上 nu 的常数基线 0.0504 → 类别查表 0.0424，绝对量本来就小，
  **nu 这一路可能整体信息量很低**，这也要一并判。

- **运维：DDP 崩了要手动收尸**（05 号票现场吃到）。rank 0 抛异常退出后，
  rank 1/2/3 **不跟着退**，变成 ppid=1 的孤儿，每个占着约 21 GB 显存不放
  ⇒ 下一次启动必 OOM，而报错指向新进程，很容易误判成"配置吃显存"。
  排查靠 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 对 `ps` 查 ppid=1。
  值不值得做成 launcher 的一道自动检查还没想清楚，先记在这里。

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
  （glob + `!` 取反，一个循环、零新增分支、两道护栏原样存活）——**2026-09-07 已实现并关闭**。
  **不重开 freeze-contract**——它的终点（验收层）确实建成了，且已在该图 Out of scope
  留了指回本票的一行。**这个口只开这么大**：除"表达'除了'"之外的任何机制改动仍在本图之外。
  **2026-09-04 更正**：原文把"bf16 静默冻结"也算进这条，是错的——它 `requires_grad` 全程为真、
  参数进了优化器，只占冻结三判据的第三条，**不是冻结**。freeze-contract 图已据此把它
  整票判出 scope（该图 07 号票），本图不能再把它推回去，否则两张图互指、它谁都不归。
  现寄存为本图 [06 号票](issues/06-aggregator-precision.md)（明确标注不在路上、不阻塞任何票）。
- **类别预测**（2026-09-07 由 [05 号票](issues/05-training-operating-point.md)划出）：
  joint 配方是 `class_agnostic: true`，`loss_segvggt.py:274-277` 把 200+1 的头 logsumexp 成
  2 列（object / no-object）、`:194-196` 把 GT 类别全填 0 ⇒ **本图的 checkpoint 只预测 objectness**。
  要真做类别得换头（Infinigen 213 类 vs 预训练 ScanNet200），是一个新的随机初始化大模块，
  config 注释自己写了 "no classifier surgery" —— 这是有意的 Phase-1 配方。
  **连带**：Not yet specified 里 `P̂ = LUT[ĉ] + Δ(q)` 依赖闭集 `ĉ`，因此它挂在"哪天做类别"上，
  在本图内不可能成立。

- **IGGT 路线**（`phys_iggt.yaml`）：aggregator/part_adaptor/part_head 全冻，phys head 是纯 frozen
  probe；要修得先给 VGGT aggregator 加 adapter，成本高一个量级。降级为"冻结底座"那一行的对照组。
- **复现 PIXIE / VoMP / PhysGS，MPM 灵敏度实验，HILO / PixieVerse**（memory `paper-direction-scene-phys`
  明令不碰）。

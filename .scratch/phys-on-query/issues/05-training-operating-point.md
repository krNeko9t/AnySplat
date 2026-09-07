# 05 — 三周内能跑完的运行点

Type: task
Status: closed (2026-09-07)
Blocked by: —  (01、02 均已关闭；(b) 臂的 config 另需 09)
Assignee: @krNeko9t (session 2026-09-07)

> ⚠️ **开跑前先看 map 的「机器」一节**：本机所有 python 都要带 `PYTHONNOUSERSITE=1`，
> 否则 user-site 的 torch 2.7.1 顶掉 env 的 2.4.1，import 阶段就炸
> （2026-09-07 [11 号票](11-geo-drift-readonly-metric.md) 现场发现）。

## Question

距 2026-09-25 只剩三周，且这个 checkpoint 必须留出重跑一两次的余地。要定：

1. 硬件：几张什么卡、能连续占多久。
2. 数据量：用 Infinigen 全量还是子集；每个场景多少视角。
3. query 数 Q、batch、步数——要能估出一次完整训练的墙钟时间。
4. **课程**：GroupForward 的教训是"先几何 → 再实例 → 再联合"，去掉 curriculum 是它全表最大的
   单项退化（mIoU 0.5989→0.4669，PSNR 26.23→18.92）。我们的起点是 pretrained SegVGGT，
   等于前两阶段已经完成 ⇒ 大概率只需要"联合"这一段，但要确认。
5. **几何保险丝**：`segvggt_geo_supervision: teacher`（frozen VGGT 蒸 depth/camera）要不要开。
   当前 joint 配置用的是 `gt` 且 `segvggt_geo.weight: 0.0` = 没有任何几何约束，
   而 01 修完之后梯度真的会进主干 —— 这个保险丝的必要性随之上升。

## 2026-09-07 更新（[01 号票](01-unfreeze-lora.md)关闭后）

**阻塞解除。但本票的形状变了：要给两臂定运行点，不是一臂。**
01 号票判的是 **(d) 两臂并行对照**——4 卡跑全冻底座、4 卡跑放开 LoRA，同时起，
唯一变量是一条 `!*.lora.*`（+9.44M / +1.9%）。

01 号票已经交掉了本票原来最难的两项，别重新推导：

- **第 1 项（硬件）与第 3 项（墙钟）已有实测底**：`bms-39468022-001`，8×A100-40G，
  2026-09-07 核实 0 MiB 占用，env `anysplat`。2026-07-28 那次 joint 跑实测
  **1.30 s/step**（8 卡 / bs=1 / 4 视角 / 252×448）⇒ 20k step ≈ **7.2 小时**。
  两臂各占 4 卡 ⇒ 每臂约 **14.4 小时**，一天内两条曲线同时到手。
  距 09-25 还有 19 天，**算力不是约束**——本票不必再为省算力做取舍。
- **第 5 项（几何保险丝）已判**：**不开 teacher，`segvggt_geo.weight` 保持 0**。
  01 号票 Q2 判定几何不作为本图交付（终点只写 class + P），代之以
  [11 号票](11-geo-drift-readonly-metric.md) 的**只读** depth/pose 漂移指标。
  本票不要把这条推回去。

**本票仍要定的**：第 2 项（数据——scannet100 100 条 vs Infinigen 全量 1466 条，
每场景多少视角）、第 3 项的 Q/batch/步数、第 4 项（课程）。
**两臂必须共享这些设置**，否则 01 号票整个对照作废。

⚠️ **开跑前的硬门槛**：任何 `freeze_keywords` 改动都要重生成
`config/experiment/locks/<X>.lock`，缺 lock 或 diff 不符 = 启动硬错
（`freeze-contract` 图已落地）。(b) 臂那份 lock 的 diff 应**恰好 192 行**从 `-` 翻成 `T`。

**完成判据**：**两份**能直接启动的配置（(a) 臂可今天就绪，(b) 臂等 [09 号票](09-freeze-matching-language.md)）
+ 各自的 lock + 一个墙钟时间估计。

---

## Answer（2026-09-07 实做）

**一句话**：两臂运行点全部定死，(a) 臂已在 4 卡上跑起来；(b) 臂 config 写好但等 09。
过程中查出**四件"不改就读不出东西"的事**，都就地改了。

### 0. 票面第 3 项自动答完：Q 不是自由参数

`aggregator.instance_query_token` 在 `segvggt_scannet200.pt` 里就是 **`(1, 400, 1024)` 的
learned Parameter**。改 Q ⇒ 形状不匹配 ⇒ `strict=False` 静默丢弃 ⇒ 整个 query bank 随机初始化，
预训练实例路径全废。**Q=400 锁死**（对照：每场景约 35 个标注实例，400 远够用）。

### 1. 定下的运行点（两臂共享，改一处必须改两处）

| | 值 | 依据 |
|---|---|---|
| 数据 | Infinigen 全量，**1387 训 / 74 验** | 见 §2 |
| 划分 | 按 `scene_XXX` 生成分片整组切，seed 42，实际 5.07% / 9 组 | 见 §2、§4.1 |
| 视角 | **4，钉死** | 见 §4.2 |
| 分辨率 | **252×448，钉死**；对 Infinigen 原生 288×512 是无损缩放（同为 0.5625） | 见 §4.2 |
| Q | 400 | §0 |
| 每步场景数 | `floor(max_img_per_gpu / views)` = `floor(8/4)` = **2/卡** ⇒ 4 卡 **8 场景/step** | §4.2 |
| 步数 | **20k**（= 160k 场景样本 ≈ 对 1387 个场景过 115 遍） | 不预支：两臂约 6 小时到手，看曲线再决定续不续 |
| 课程 | **无** | §3 |
| 几何 | 不监督（`weight: 0`、`teacher` 不开），只留 11 号票的只读 `val/geo_*` | 01 号票已判，本票不推回 |
| lr | `1e-4` / `query_physgm ×5 = 5e-4`（维持现状） | §5 |
| val | `limit_val_batches: 1.0`（全量 74 场景）、`val_check_interval: 500` | §4.4 |

产物：`config/experiment/phys_query_arm_a_frozen.yaml` +
`config/experiment/locks/phys_query_arm_a_frozen.lock`（与
`segvggt_agnostic_phys_joint.lock` **逐字节相同**，只差抬头——两者本就只差数据与调度，
1088 行可训，与地图记的数字对上）、`config/experiment/phys_query_arm_b_lora.yaml`（不可跑，见 §6）。

### 2. 第 2 项（数据）：Infinigen 全量，不是 scannet100

**两份数据其实都有 VLM 标签**，票面没写这一点：

| | 场景 | 帧 | 原始分辨率 | 标签 |
|---|---|---|---|---|
| `manifest_phys_scannet100.jsonl` | 100 | 83,818（中位 776/场景）| 690×920 | `instascene_preprocessed_data_zxl/outputs/run_scannet_100` |
| `$R/manifest_infinigen_phys.jsonl`（02 产出）| 1466→**1461** | 146,034 | 288×512 | `preprocessed/annotations/infinigen/` |

选 Infinigen 的**决定性理由不是数据量，是参照系**：目的地是"学生逼近教师"，
而"教师"这条线的全部刻度（04 号票：类内 73.6%、查表 vs 常数 −15.6%、grounding 49.3%）
**只在 Infinigen 上存在**。scannet100 上跑出的 checkpoint 主表没有横坐标；
而且 100 个场景要吃 160k 样本 = 每场景 1600 遍，几乎必然过拟合。
代价（合成数据 vs ScanNet200 预训练的域差）是真的，但**同等作用在两臂上，不污染 01 的对照**。

⚠️ 仓库里的 `manifests/manifest_phys_infinigen.jsonl`（1466 行 / 32 帧一场景 / 绝对路径）
是 09-04 的旧件，**不是 02 的产物，别用**。要用 `$R/manifest_infinigen_phys.jsonl`。

### 3. 第 4 项（课程）：不做

GroupForward 的 curriculum 收益来自"让几何先立起来"，而我们的几何是**冻结的预训练几何，
从第 0 步就立着**；唯一随机初始化的是 `query_physgm`(~0.2M)，它已有独立的 5× lr 组在做同一件事。
另外**仓库里根本没有 curriculum 机制**（全仓 grep 只有 `instseg_inscene_infinigen_mv.yaml:48`
一句注释），分段只能手工跑两次，给两臂各引入一个 ckpt 交接面——纯风险。

### 4. 四件"不改就读不出东西"的事（本票就地改了）

#### 4.1 训练与验证此前共用同一份 dataset cfg ⇒ **val 跑在训练场景上**

`data_module.py:121` 与 `:180` 传的都是 `self.dataset_cfgs`，`DatasetManifest` 只按 `stage`
改 augment/log，不筛场景。而 04 号票已判"213 类里 126 个单例 ⇒ 只能按场景划"。
不补，两臂对照的唯一读数（曲线）就没得比。

改：`DatasetManifestCfg.scene_split_path` 指一份
`{"train_scene_ids": [...], "val_scene_ids": [...]}`；`scripts/split_scenes.py` 生成。
**按 `scene_XXX` 整组切**——理由是防资产泄漏，但**实测该风险不存在**：抽 300 对算物体名
Jaccard，**组内 0.049 vs 组间 0.047**，一个组里两个场景的实例名精确重合是 48 个里 1 个
⇒ `scene_XXX` 只是生成任务的分片。分组切因此是免费的保险，照做。
结果 `config/experiment/splits/infinigen_phys_split.json`：**1387 / 74（5.07%，9 组）**（1461 个可用场景，见 §9）。

#### 4.2 `DynamicBatchSampler` 有两个没人提过的抖动

`data_sampler.py:318` 传的是 `[2, num_context_views]`，权重 ∝ n²
⇒ **每样本视角数随机 ∈ {2,3,4}**，p = 4/29, 9/29, 16/29，期望 3.41。
`h_range` 直接就是 `input_image_shape` ⇒ `ps_h = randint(252//14, 448//14)`
⇒ **训练时 H ∈ [252,448]、W 恒 448**；而 `rescale_and_crop` 用
`scale_factor = max(h_out/h_in, w_out/w_in)` + center crop
⇒ 对 288×512 的 Infinigen，H=448 那一端要放大 1.556 倍再横向裁到 448，**只剩 56% 的画幅宽度**。
val 则固定在 252×448。

⇒ 所谓"252×448"从来不是训练分辨率，是**训练区间的下端 + val 的固定值**；
`data_loader.train.batch_size` 在这条路径上**是死的**（`data_sampler.py:118` 明写）。

钉死的理由：抖动虽两臂同序列、不破坏对照，却把两条曲线的方差一起放大，而
20k step 的两臂差值本来就不一定大；且"裁掉 44% 画幅"是往 04 号票刚证明的最大伤口
（grounding 看错物体）上撒盐，而 07-28 那次 scannet 基线（0.75 宽高比，只裁 25%）替它背书不了。
改法：`fixed_views_and_shape: true` 给采样器传**退化区间**（`[4,4]` / `[252,252]`），
**采样器内部零分支**，抖动路径一行没动。

#### 4.3 z-score 常数是抄来的（07 号票就地解决了大半）

07 号票原本阻塞于 03，因为 train-only 拟合要等划分。**4.1 一落地这个阻塞当场消失**，
所以本票顺手拟合并换掉（用错常数跑 7 小时再重跑，会吃掉"重跑一两次的余地"里的一次）：

| 量 | 新（训练集拟合） | 旧（PhysGM 逐字抄） |
|---|---|---|
| density (log10 kg/m³) | 2.863740 / 0.399147 | 3.0 / 0.5 |
| youngs_modulus (log10 Pa) | 9.495947 / 1.317972 | 7.387210 / 2.456477 |
| poisson_ratio (raw) | 0.336525 / 0.066235 | 0.398 / 0.111 |

1387 场景 / 49,214 实例，与 02 号票的全语料拟合值**小数点后 3 位一致**（留出集没带偏）。
**落在哪**：直接改死 `parsers.py` 的 tuple（本图只有一份语料，config 指 JSON 买不到东西，
却买来一个"推理期读了另一份"的静默错误面；07 号票自己写了"改一处即可，别改成两处"），
注释里写明拟合来源与日期，凭据 `config/experiment/splits/physgm_norm_infinigen_train.json` 进 git。
⚠️ **副作用**：`physgm_iggt` / `physgm_mvimgnet2` / `segvggt_physgm` /
`segvggt_agnostic_phys_joint` 等所有用 `instascene_vlm_physgm` 的配方，
归一化都跟着换了。本图不跑它们，但**谁要拿旧 ckpt 做推理，反归一化对不上**。

#### 4.4 val 里此前**一个物性数都没有**

`segvggt_wrapper.py` 的 val 指标只有 `queries_fired` / `depth_mean` / `geo_*` /
实例那一组。加上 `assert b == 1` 且 `limit_val_batches: 1` ⇒ 每次 val 只看 4 个场景。
⇒ 照原样跑，两臂的 40 次 val 里关于 P 一个字都没有，只能读 Gaussian NLL——
而 03 号票已写明它不适合当报告指标。

新增 `src/evaluation/physics_metrics.py`，与 11 号票的 `_log_geo_drift` 同构
（`@no_grad`、不进 loss、指标在 wrapper 里不在 config 里 ⇒ **两臂自动都有**）：

- **三行同批对照**：学生 / 常数（预测训练均值，即 z=0）/ **类别查表**
  （预测该实例类别在训练集上的均值）。教师**故意不是一行**——学生的标签就是教师，误差恒 0。
- **匹配复用 `instance_metrics` 的 IoU 最优指派**（直接 import，不重写）
  ⇒ 物性数与 `val/matched_iou_mean` 描述的是**同一批实例**，包括训练早期偏向大而易的物体这个偏差。
- **误差直接由 `std·|Δz|` 得到**：z-score 的 std 是每个量固定的常数，所以
  `|x_pred − x_gt| = std·|z_pred − z_gt|`，不用反归一化往返、不引入 clamp 不对称。
  E/ρ 报 **log10 SI MAE**，ν 报 **raw MAE**。
- 类别查表那一行由 `scripts/build_class_mean_lut.py` 预先 join 好
  （`$R/physgm_class_mean_z.json`，**类均值只在 1387 个训练场景上拟合**，210/213 类，
  仅 3 个实例回落到全局均值），挂在 `PhysGMTarget.class_mean_lut` 上，
  按约定从 dataset root 发现，缺文件就静默无该行，**任何 loss 都不读它**。

⚠️ **控制台那行只是第一个 val batch（一个场景）**：`logger.info` 的条件是
`batch_idx % 100 == 0`（与既有的实例指标那行同构）。**进 tensorboard 的 `val/phys_*` 才是
74 个场景的聚合值**，读曲线要看后者，别拿日志里的单场景数下结论。

**验收（全量 74 个 val 场景 / 2724 个实例，离线）**：

| 量 | 常数 | 类别查表 | |
|---|---|---|---|
| log10 density | 0.2708 | 0.2260 | −16.6% |
| **log10 youngs_modulus** | **0.9050** | **0.7477** | **−17.4%** |
| poisson_ratio | 0.0504 | 0.0424 | −16.0% |

与 04 号票的 **−15.6%（0.890→0.751）一致**。⚠️ **分片间波动不小**：换一份 9 组的划分
（同脚本、min_views 修正前）给的是 −12.5%，所以这个数别当三位有效数字用。
⇒ **trivial baseline 确实很弱这条独立复现了**，而且这次是留出集上的数，可以进论文。
（注：MAE 由中位数最小化而非均值，类均值 LUT 因此略吃亏；要不要改成类中位数归 03。）

### 5. lr 维持 `1e-4 / 5e-4`（10 号票的答案由这次跑给出）

10 号票的推理线指向 `2e-5 / 1e-4`。**但它没看到一个耦合**：(b) 臂放开的是 LoRA，
而 01 号票明写"LoRA 用默认组 1e-4——为保住单变量"。把默认组降到 2e-5，那 9.44M 也跟着降，
**LoRA 在 2e-5 上 20k step 很可能几乎不动**——那时读到的"两臂没差别"是 lr 太冷造成的，
不是 01 号票问的问题。本次跑的产物就是这个对照，不能让未定的 lr 把它读成噪声。
⇒ **10 号票改成"从两臂曲线读答案"**。代价说清楚：若 1e-4 真烧坏了 454M 预训练参数，
两臂会一起烂，7 小时白花且分不清病因。

### 6. (b) 臂：config 写好，**不可跑**

`freeze_keywords` 是裸子串 OR（`base_wrapper.py:515`），写不出"block 里除了 lora 全冻"。
(b) 臂的 config 已按 09 号票将落地的语法（glob + `!` 取反）写好并标了醒目的阻塞头，
**没有生成 lock**（现在生成会是错的）。09 落地后：重生成 lock，
**diff 应恰好 192 行**从 `-` 翻成 `T`。

**没有等 (b) 一起点火**（01 号票的"同时起"理由是省墙钟，不是对照的必要条件）：
两臂共享全部设置与同一份划分，隔一天起不影响可比性；而早一天拿到 (a) 臂曲线有独立价值——
它先回答"§5 的 1e-4 有没有烧坏东西"和"§4.4 新指标是不是真能动"，这两件 (b) 臂都要依赖。

### 7. 顺手看到的、留给别的票的东西

- **`class_agnostic: true` ⇒ 这次跑出来的 checkpoint 不预测类别，只预测 objectness。**
  `loss_segvggt.py:274-277` 把 200+1 的头 logsumexp 成 2 列，`:194-196` 把 GT 类别全填 0。
  这是有意的 Phase-1 配方（config 注释自己写了 "no classifier surgery"），
  但**与目的地"query 在预测 class 之外额外输出 P"的措辞冲突**。
  ⇒ 已改地图的目的地措辞，类别预测明确出 scope（见地图 Out of scope）。
- **`limit_val_batches: 1` 是个陷阱**：`validation_step` 有 `assert b == 1`，
  所以它等于"每卡一个场景"。本票改成 `1.0`（全量），74 个场景的 val 只花几秒。

### 8. 复现

```bash
conda activate anysplat && export PYTHONNOUSERSITE=1
R=/mnt/storage_pool/liaoyuanjun/data/InsScene-15K
python scripts/split_scenes.py --manifest $R/manifest_infinigen_phys.jsonl \
  --out config/experiment/splits/infinigen_phys_split.json
python scripts/fit_physgm_norm.py --root $R --manifest $R/manifest_infinigen_phys.jsonl \
  --scene_ids config/experiment/splits/infinigen_phys_train_ids.txt \
  --out config/experiment/splits/physgm_norm_infinigen_train.json
python scripts/build_class_mean_lut.py --root $R --manifest $R/manifest_infinigen_phys.jsonl \
  --class_table $R/infinigen_class_table.json \
  --split config/experiment/splits/infinigen_phys_split.json \
  --out $R/physgm_class_mean_z.json
python scripts/freeze_lock.py +experiment=phys_query_arm_a_frozen
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py +experiment=phys_query_arm_a_frozen
```

**墙钟（本票的第 3 项）**：(a) 臂实测 **≈1.1 s/step**（4×A100 / 4 视角 / 252×448 /
每卡 2 场景，前 90 步）⇒ 20k step ≈ **6.1 小时**，比 scannet100 那次的 1.30 s/step 略快
（Infinigen 图更小，解码便宜）。**19 天里两臂各跑一次只花一个上午**，算力确实不是约束。

**启动核对**（4 卡日志）：`frozen 1533 params total`、
`param_groups[0] keywords=['query_physgm'] -> 15 params` + `default -> 1073 params`、
**`Trainable params: 487 M`** —— 与地图记的 1533/1088/487M 全部对上，lock 校验通过。
⚠️ 日志里 LoRA 自己打的 `Trainable percentage: 85.32%` 是**构造期、`apply_freeze()` 之前**的数，
不是冻结后的可训比例，别被它吓到。

提交 `40ff919` / `3a14983` / `005ed62` / `8b3c4b6`。

### 9. 追修：视角钉死暴露的"帧数不够"崩溃（提交 `3a14983`）

(a) 臂第一次点火在 step ~100 炸了：
`ValueError: Example does not have enough frames!`（dataloader worker 内）。

**是潜伏 bug，不是本票引入的。** `DatasetManifest` 那个"视角不够就丢场景"的过滤器
（注释自称就是为了防这个 exception）**算错了界**：它比 `num_context_views`，
而 bounded/bounded_fixed 真正要的是**间隔**——`num_ctxt_gap_mapping[4]` 给
`min_gap=12`，`max_gap` 又被 clamp 到 `num_views-1`，所以**需要 13 帧，不是 4 帧**。

抖动路径下每样本视角数随机 ∈ {2,3,4}，短场景**只在抽到 4 的时候炸**（16/29 的概率）
⇒ 以前表现为随机崩，钉死之后变成必炸才被抓到。scannet100 每场景 ≥272 帧，从没碰上过。

改在源头：两个 sampler 加 `min_frames_required`（取所有可能 `num_ctxt` 里最大的
`min_gap + 1`——短场景"有时炸有时不炸"比"被丢掉"更糟），`DatasetManifest` 问它要界。
Infinigen 里够不着的场景共 5 个（1/1/2/5/10 帧）⇒ **1466 → 1461**，划分与常数随之重算
（上表已是修正后的数）。

**教训记一笔**：`fixed_views_and_shape` 这类"把随机变量钉成常数"的改动，
会把概率性故障变成确定性故障——这次是好事（抓到了），但下次同类改动要预期
"以前偶发的东西现在必现"，别误判成新引入的 bug。

### 10. 追修二：DDP 崩了不会自己收尸（运维，非代码）

修完 §9 重新点火，四张卡里三张立刻 OOM。**不是配置吃显存**：
`nvidia-smi --query-compute-apps` 显示 §9 那次崩溃的 **rank 1/2/3 还活着**
（pid 2690846/47/48，`ppid=1` 的孤儿），每个占着约 **21 GB** 不放，外加一大群 dataloader worker。

**rank 0 抛异常退出后，其余 rank 不跟着退。** 报错却指向新起的进程，很容易误判成
"新配置吃显存"。排查手法：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
拿到 pid，再 `ps -eo pid,ppid,cmd` 看谁的 ppid 是 1。收尸后 8 张卡回到 0 MiB，第三次点火正常。

⚠️ `pkill -9 -f <pattern>` 会匹配到自己那条 shell 命令行并自杀，用显式 pid `kill -9`。

这条已记进地图的 **Not yet specified**（要不要做成 launcher 的自动检查还没想清楚）。


### 11. 起跑后的头两个读数（不是结论，只是"指标活着"的证据）

| step | 学生 log10 E | 常数 | 类别查表 |
|---|---|---|---|
| 0 | 0.7872 | 0.7825 | 0.9575 |
| 500 | 1.0837 | 0.9010 | 0.9521 |

（都是控制台那行 ⇒ **单个 val 场景**，见上面的 ⚠️。）

两点值得先记下：

1. **step 0 学生 ≈ 常数基线，这是构造决定的**：头随机初始化时 `mu ≈ 0`，而 z=0 **就是**训练集均值。
   ⇒ "学生打不过常数"从第 0 步起就是一条有意义的警戒线，不需要额外定义。
2. **step 500 学生反而退到常数之后**（1.08 vs 0.90）。此时 `warm_up_steps: 1000` 还没走完、
   `query_physgm` 在 5e-4 上，早期乱走属正常，**但这正是 [10 号票](10-joint-lr.md)要盯的那条线**：
   若到几千步还回不到常数基线以下，就是 lr 太烫的证据。


### 12. 追修三：根盘写满（`OSError: [Errno 28]`）

(a) 臂在 **step 1990** 死于写 checkpoint 时磁盘满，并再次留下孤儿 rank（同 §10）。

**账**：`save_weights_only: false` ⇒ 一份 ckpt **9.8 G**（1.37B fp32 权重 5.5 G +
487M 可训参数的 AdamW 状态 3.9 G）。`save_top_k: 1` 之外 Lightning 另留 `last.ckpt`
⇒ **每臂常驻约 20 G**，写新的那一刻峰值近 30 G。而仓库所在的 `/dev/sda1` 是 439 G、
长期 **93% 满**，可用只剩 685 M。两臂一起跑必然写满。

**改**：两份 config 的 `hydra.run.dir` 改到
`/mnt/storage_pool/liaoyuanjun/runs/exp_${wandb.name}/...`（3.5 T 阵列，766 G 可用，
数据本来就在那），理由写进 config 注释。删掉本 session 三次崩溃跑留下的 30 G 输出后根盘回到 31 G 可用。

**这条已升进地图的「机器」一节**——它不是本票的事，是这台机器上所有训练的事。

**监控的教训**：我第一版 `Monitor` 的 grep 里没有 `OSError` / `No space`，
只靠 `Traceback` 兜住（这次兜住了，但纯属运气——磁盘满也可能表现为静默的 ckpt 缺失）。
过滤器要按"这个进程现在崩了，我这条 grep 会不会出声"来设计，不是按"我期待看到什么"。

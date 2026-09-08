# Map: 把物性头长到 IGGT 上，端到端出「实例 + 物性」

Label: wayfinder:map

## Destination

输入**一个已经建好的 3DGS 场景 + 它自己的相机 / 2D 观察**，一次前馈，在 3D 空间里得到
**实例分割**（IGGT 官方权重 + HDBSCAN）+ **每个实例挂着 (E, ν, ρ)**（我们自己训的物性头）。

交付：`3dovs/bench` 上**一张全分辨率实例叠加图 + 一份实例物性表 + 可复现脚本**；
同一套脚本在 `mipnerf360/garden` 上**不崩**。

**判据是主观并排验收**：和官方 IGGT 权重自己的 trace 结果并排看**不明显更差**，
且每个实例都有 (E, ν, ρ) 的值。不设 IoU 硬阈值——本图没有能支撑硬阈值的依据
（见 Out of scope 里作废的 10/11 号票）。

**真边界是导师，日期是 2026-09-11（后天）。** 他要看到的是「这件事被完成」：
3DGS 场景进 → 一次前馈 → 一堆带物理参数的 3D 实例出，且链路里至少有一次我们自己完成的训练。
**论文主张后定**——等结果出来看什么对论文有利再反过来定，因此**不进 Destination**。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）。票可以是动手的。

**领域**：前馈多视角场景理解 → 已有高斯场景的 3D 实例分割 + 实例级物性。
**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

**票的粒度与节奏**（2026-09-09 用户校准，我此前的时间估计一直偏保守）：
**一张票约半小时**，因为票基本都交给 agent 做；**训练是唯一实打实的时间**。
⇒ **关键路径 = 物性头的训练**。01 → 02 今天做完、今晚起跑，
03 / 04 在训练跑的时候并行，05 收口。

**环境**（继承 [phys-on-query 图](../phys-on-query/map.md)，不重新核）：
GPU 服务器 `bms-39468022-001`（8×A100-40G），conda env `anysplat`，
⚠️ **跑任何 python 都要带 `PYTHONNOUSERSITE=1`**（user-site 里的 torch 2.7.1+cu118 会顶掉 env 的 2.4.1+cu124）。
⚠️ **训练输出写 `/mnt/storage_pool/liaoyuanjun/runs/`，不写仓库下的 `output/`**（根盘 439G 长期 93% 满）。
两个 rasterizer 都已装：`diff_gaussian_rasterization`（3DGS）和 `diff_surfel_rasterization`（2DGS）。

### 本图开工前已定的事（2026-09-09 grilling，不再重开）

1. **分割 = 官方 IGGT 权重，完全不训**（`/home/liaoyuanjun/iggt_checkpoint.pth`）。
   它就是用户说「目前能看」的那份，是本图唯一一个**已知能看**的东西，不能把它也变成变量。
   **不留「不够看就接着训」的口子**——留了它，两天里迟早有一天会去动它。
2. **物性 = 新头，我们训一次，frozen probe 起步**（aggregator / camera_head / point_head /
   depth_head / part_adaptor / part_head 全冻）。
   ⚠️ [phys-on-query 图](../phys-on-query/map.md) 当初把 IGGT 路线判出 scope 的理由是
   「物性头是纯 frozen probe，要修得先给 VGGT aggregator 加 adapter，成本高一个量级」。
   **这个顾虑在本图的判据下是消解的，不是被忽略的**：用户口径是「有个值就行」，frozen probe 恰恰够。
3. **物性不承诺任何量化声明**。不重训、不改头之外的东西，不报物性主表。
4. **场景 = bench 主 + garden 只证不崩。** alocasia / PhysicsDreamer 那批不取（数据在失联的
   `/mnt/shared-storage-gpfs2` 上）。
5. **论文主张后定**，不写进 Destination。
6. **SegVGGT 那条线整条搁置**，不在本图内做任何后续。

### 事实底座（2026-09-09 现场读码，是本图的地基）

**G1 — IGGT 权重在本地，路径也已经是本地的。**
`/home/liaoyuanjun/iggt_checkpoint.pth` 实在；`physgm_iggt.yaml:27` 与
`physgm_dpt_iggt.yaml:33` 里写的就是这个本地路径（**不是**失联的 `/mnt/shared-storage-gpfs2`——
只有 `phys_iggt.yaml:26` 还指着失联挂载点）。回 IGGT 不被挂载点卡住。

**G2 — IGGT 上的物性头不是白纸，有四个 scheme（`iggt.py:54`），但要分清两类形态。**
- `physgm_copy`（`physgm_iggt.yaml`）：池的是**整帧 aggregator token** ⇒ 一帧一个值，
  **scene-level，不满足「实例物性」**。
- `physgm_dpt`（`physgm_dpt_iggt.yaml`）：DPT dense feature map（image 分辨率，
  `phys_feat_dim: 32`）按 **instance_mask 掩码平均池化** ⇒ **实例级**，是本图的候选。
- `class` / `property`：同样吃 `instance_mask`，但输出的是 4 类 / 属性向量，不是 PhysGM 的
  (mu, var) 口径，与已有的归一化常数和 loss 对不上。
⇒ **「新加物性 head」实际是「选 `physgm_dpt` + 改它的推理侧」，不是从零写头。**

**G3 — 训练/推理失配是这条路自带的。**
`PhysicsSchemeInputs.instance_mask`（`scheme.py:43`）在训练时来自 **GT**。
SegVGGT 的 query 自己就是实例，这个问题不存在；**IGGT 没有 query**，实例是 HDBSCAN 事后聚出来的。
这是本图相对 SegVGGT 路线**唯一真正新增的风险**。

**G4 — 有一条把 G3 化掉的路（01 号票要判，不是已定的事）。**
`physgm_dpt` 的头是「dense feat → 掩码平均 → per-property MLP」，
**池化严格在 MLP 之前**（`physgm_dense_readout.py:68-90` 逐行确认：
`pool_one_sample(feat_map[b], instance_mask[b])` → `pooled [K,C]` → `decoder(pooled)`）。
⇒ 推理时可以把 dense feat **trace 到高斯上**，按 HDBSCAN 实例**在 3D 里平均**，
再过**同一个 MLP**——头一个字节都不用改，失配只剩「GT mask vs HDBSCAN 实例」
和「2D 逐像素池 vs 3D 逐高斯池」两条，都是可量的。

**G5 — 老图 02 号票的分趟 trace 是本图的前提，不是白做的。**
`phys_feat_dim: 32` > `TRACE_CHANNELS = 20`（`src/trace_render/trace_rasterize.py:20`）
⇒ 物性特征要 2 趟；连同 8 维 instance feat 一起 3 趟。不分趟根本搬不动。

**G6 — 训练接线是抄的，不是新写的。**
`phys_query_arm_b_lora.yaml:149-161` 的 dataset 块可直接搬：infinigen manifest
（`/mnt/storage_pool/liaoyuanjun/data/InsScene-15K/manifest_infinigen_phys.jsonl`，1466 行 ⇒
1387 训 / 74 验，`config/experiment/splits/infinigen_phys_split.json`）、
`instascene_vlm_physgm` parser、252×448、4 视角、`fixed_views_and_shape: true`。
`PHYSGM_NORMALIZATION`（`parsers.py:74`）已经是**本语料训练集拟合值**
（density 2.863740/0.399147、E 9.495947/1.317972、ν 0.336525/0.066235）。
⚠️ `physgm_dpt_iggt.yaml` 现在的 dataset 指的是 **scannet100**，要换。

**G7 — HDBSCAN 已实现且自带 subsample，但代价从没实测过。**
`src/instseg/hdbscan_assign.py`：subsample → HDBSCAN → NearestCentroid 全量赋值。
用户点名要重新评估这个代价 ⇒ 03 号票量它。


**G8 — ⚠️ IGGT 的实例特征空间是 batch-relative 的（2026-09-09 由 03 号票发现，地图原先没有这条）。**
同样 4 个视角，单独跑 / 放进 8 张的 batch / 放进 48 张的 batch，逐像素实例特征会变：
平均 cosine 0.958 / 0.947，**5 分位 0.76、最小 0.19**。
控制组：同一分组跑两遍 cosine = 1.0000 精确 ⇒ **编码器是确定性的，漂移全部来自 batch 组成。**
bench 碰不到（36 视角一次前馈）。**garden 的 185 视角被切成 4 个 48 的 batch，
全场景 HDBSCAN 因而混了 4 个朝向不同的特征空间**，而漂移量（≈0.05）与
`cluster_selection_epsilon`（0.06）同量级。实测 garden 的主要簇扛住了，
但**任何视角数多于一次前馈的场景都踩着这个雷**。

**G9 — `sam/mask` 不是 3D 实例 GT，是逐帧 SAM 输出。**
帧 00 是 id 1–8、帧 12 是 1–7、帧 24 是 1–9，**同一 id 在不同帧指不同物体**。
⇒ 老图记的「bench 8 个实例」是**帧 00 的计数**；04 号票报的「10 个」是 36 帧 id 并集的最大值，
**两个都不是场景实例数**。**在原始 id 上直接算 IoU/mIoU 是错的。**
bench 的 stuff 像素占比实测 **84.2%**（不是地图原先写的 62%），stuff 问题比假设的更重。

⚠️ **G9 的结论后被 06 号票软化（观察不变）**：`sam/mask` **按原样**不是 3D GT，
但经**逐 `(frame,id)` 反投影 + 跨帧高斯集合重叠连接**是**可以造出一个**的。
06 实测：285 个 `(frame,id)` 节点连出**恰好 7 个跨满 36 帧、零矛盾的分量**
（同一分量从不在一帧里占两个 id），**τ ∈ [0.3, 0.7] 上逐位相同**，
划走被 trace 高斯的 **99.79%**。
⇒ **bench 有 7 个帧一致的 3D GT 实例（2 stuff + 5 物体）。**
⚠️ 甲侧划分要**按 co-visibility 归一**（按原始帧数归一会把墙从 541k 错误缩到 274k）。

**G10 — trace 的两个坑，都由 06 号票踩出来。**
1. ⚠️ **`gau_phys_feat` 是物性头的函数**——04 号票的 `.pt` 是 **step 500** traced 的，
   换 checkpoint 就必须重 trace。（`num_ray` 与 `gau_inst_feat` 不受影响：
   重 trace 后 `num_ray` 逐位相同、`gau_inst_feat` 差 3.4e-07 ⇒ 几何可证相同，
   **03 的 HDBSCAN 标签可跨 checkpoint 复用**。）
2. ⚠️ **复现 04 的几何需要 `--no_trace_crop_aligned`**——默认 `True` 给 95.9% 覆盖率而非 96.22%，
   **且这个 flag 没有记进 `.pt` 的 provenance**。

**G11 — 32 维物性特征在训练中长大约 1060×**（|x|max 0.485 @step500 → 515.2 @step10000）。
**只因为 decoder 首层是 LayerNorm 才无害**——即 01 号票更正块认定的那条承重事实。
任何绕过或去掉那个 LN 的改动都会立刻炸。

### 从两张老图继承的东西（引用，不重抄）

- [老图 05 — 相机对齐](../segvggt-trace-3dgs/issues/05-preprocess-camera-alignment.md)：
  bench 36 帧同尺寸横图 ⇒ 整图 resize 对 `TraceCamera` 是**无操作**，
  `trace_cams = list(cam_list)` 直接用；`encoder_batch_size = 4`。
- [老图 02 — 分趟 trace](../segvggt-trace-3dgs/issues/02-chunked-trace-over-20-channels.md)：
  `trace_single_view_chunked`，168.5 ms/view，判据脚本 `scripts/check_chunked_trace.py`。
- [老图 04 的验收教训](../segvggt-trace-3dgs/issues/04-query-pooling-and-3d-dedup.md)：
  **交付图必须是全分辨率叠加图（原图 + 半透明实例色），不是灰底 3DGS 渲染。**
  上次拿 1/4 分辨率灰底渲染（灰色占 87.8%）去验收，被判不合格，而同一份结果叠回原图边界是对的。
- **老图 F4**：trace kernel 用 `atomicAdd`，同一条路跑两遍差 maxabs 3.4e-3（相对 1.7e-6）
  ⇒ 一切「trace 前后一致」的比较都要带控制组，不能用 `torch.equal`。
- **phys-on-query**：PHYSGM 归一化常数、`src/evaluation/physics_metrics.py` 的三行对照
  （学生 / 常数 / 类别查表）、`apply_freeze()` 的 glob + `!` 取反与 lock 执法机制。

## Decisions so far

<!-- 一行一个已关闭的票 -->

- [01 — 物性怎么从 dense feat 走到 3D 实例](issues/01-physics-from-dense-feat-to-3d-instances.md)：
  走**候选 A（3D 池化，头一个字节不动）**。池化严格在 MLP 之前且 decoder 首层是 `LayerNorm`
  （吃掉 2D/3D 池化的一阶尺度差）⇒ trace 32 维 dense feat 到高斯，按 HDBSCAN 实例
  **`num_ray` 加权**平均（训练侧是「跨视角逐像素等权」`physics_pool.py:79-80`，
  `Σ num_ray·feat / Σ num_ray` 才是它的对应物），过同一个 MLP，最后才 `physgm_denormalize`。
  **32 维不降**（省 6 秒不值得重训）；**`var` 与 `mu_spread` 分两列**，`mu_spread` 定义为
  逐高斯单独 decode 出的 `mu` 的标准差。两处不等价分派：池化口径 → 04，成员集合 → 06。
  **给 02 的交待：训练侧照原样配，不改头、不改 loss、不改 `instance_mask` 来源。**

- [02 — 物性头的训练接线与运行点，并起跑](issues/02-train-physics-head.md)：
  **训练 2026-09-09 00:19 起跑，PID 500516，GPU 0/2/3/4，ETA ~02:04。**
  dataset 换 Infinigen（manifest 1466 行 / 1387 训 74 验，`PHYSGM_NORMALIZATION` 逐位对上，没动）、
  `max_steps` 60000→**10000**、`hydra.run.dir` 挪到 `/mnt/storage_pool`、
  `freeze_keywords` → `["*", "!*physics_scheme*"]`。lock：**1729 参数 134 可训，
  `physics_scheme` 之外 0 行可训**。log 在 `runs/logs/physgm_dpt_iggt_2026-09-09_00-19-50.log`。
  ⚠️ **更正一条记录**：旧的六前缀枚举**并没有**漏掉东西——这套 build 只有七个带参模块，
  没有带参数的 SamProjector / instance head，新旧写法逐位等价（`structure` sha256 未变）。
  改保留是因为 glob 写法对未来加模块是不变式。

- [04 — trace 侧同时搬 instance feat 和 physics feat](issues/04-trace-two-feature-sources.md)：
  **成了。** `runs/ticket04_iggt_phys/bench/gaussian_iggt_phys_feat.pt`（208 MB）里有
  `gau_inst_feat [N,8]`、`gau_phys_feat [N,32]`、`num_ray [N]`、**`blend_mass [N]`**，
  共享同一套下标。覆盖率 **96.22%**（老图 96.2%），3 趟 **79.2 ms/view**，不重编译 `TRACE_CHANNELS`。
  改动对旧默认路径是加性的（−2 行，都是 choices/print），数值与控制组同到三位有效数字。
  噪声底 **≈ 6e-7 相对**。「同一份高斯」由每趟对 `num_ray` 的逐位断言执法。
  **池化口径读数**：2D↔3D cos 均值 0.87，但**这是实例尺寸效应**——
  >1000 高斯的七个实例 cos 0.95–0.99，四个小/薄实例（534–1306 高斯）塌到 **0.39–0.82**；
  而**加权 vs 等权 cos = 0.996，几乎无关紧要**。

  ⚠️ **本票更正了 01 号票的推导**（结论不变）：`gau_sem` 是 **alpha 加权**累积、
  `num_ray` 是不加权计数（`trace_rasterize.py:251`/`:255-256`），
  常数 1.0 实测 `gau_sem/num_ray` p50 = 0.0124 而非 1.0、p95/p05 = 114×。
  重做代数后 `num_ray` 加权仍对且更强（`Σ num_ray·feat = Σ gau_sem = Σ_r W_r f_r`），
  且**分母因 LayerNorm 的正标量尺度不变性完全不重要**——池化的全部内容是
  「把该实例的高斯的 `gau_sem` 加起来」。承重的是 LayerNorm，不是分母的推敲。
  ⚠️ **bench 的 GT mask 是 10 个实例，不是老图记的 8 个。**

- [03 — 官方 IGGT 权重的实例基线 + HDBSCAN 代价](issues/03-iggt-baseline-and-hdbscan-cost.md)：
  **验收对照组已就位**——`runs/ticket03/overlay_bench_default/contact_sheet.png`，
  6 帧 **1008×756 原生分辨率**、原图在下半透明实例色在上、无灰底（覆盖率 100%）。
  orchestrator 逐张看过，边界贴着物体；**六视角同物体同色 ⇒ 顺带兑现了 05 的交付物 4**。
  bench **9 个实例** = 2 stuff（墙 42.6% / 地板 40.9% 像素）+ 6 个真物体 + 1 个 1,025 高斯的碎片。
  **HDBSCAN 代价：bench 9.64 s / 1.01 GB RSS，garden 11.36 s / 1.10 GB；
  代价由 `max_points` cap 决定而非场景大小**（garden 多 74% 高斯只贵 18%）。
  ⚠️ **真正的大头是 `postprocess_knn_k=20` 的 cKDTree 平滑：13.87 s / 1.78 GB，比 HDBSCAN 还贵。**
  **参数：什么都不改。** 50/10/0.06 @ 200000 是网格里唯一在两场景上都不退化的点
  （eps ≥ 0.10 两边塌、≤ 0.04 两边碎）；调高 `max_points` 反而更差（500k 上 bench 把墙和地板并了）。
  噪声底是**零标签变动**（两次独立 trace 得到完全相同的 9 个实例，ARI 1.0000）。
  ⚠️ **garden：相机/图像/trace/聚类/叠加全链路已跑通，但本机唯一的 garden PLY 是
  31.5% 的中断拷贝**（header 声明 5,834,784 顶点，实有 1,839,236；文件恰好 435 MiB 整）。
  在残缺场景上跑通不崩（185 视角、82 实例）。完整 garden 需重拷 ~1.45 GB。
  garden 另需 `scripts/make_transforms_from_3dgs_cameras.py`（它没有 COLMAP `sparse/`，
  转换约定已用渲染对齐验证）。
  ⚠️ 更正：`configs/trace/*.json` 的**场景路径**并非「全部指向失联挂载点」，
  只有 `alocasia.json` 是；真正指着失联挂载点的是 6 个里 3 个的 `iggt_model_path`。

- [06 — 量「成员集合」这处不等价](issues/06-quantify-the-two-mismatches.md)：
  **结局 (a)：GT 造得出来，失配可忽略。** bench 连出 **7 个帧一致的 3D GT 实例**
  （τ∈[0.3,0.7] 逐位稳定，覆盖被 trace 高斯的 99.79%）。
  甲（GT 成员）vs 乙（HDBSCAN 成员）过同一个 step-10000 的 MLP：
  **`mu` 移动从不超过 0.16 z = 该实例自身 `mu_spread` 的 ≤0.28 倍**，
  远在头自报的 σ（√var ≈ 0.65–1.4）内，没有实例改变材料档次，ν 从不超过 2.6%
  ⇒ **在本图判据（「有个值就行」）下可忽略。**
  ⚠️ **但 SI 上不可忽略**：E 经 `10^(1.318·z)` 解码，同样 0.16 z 在 7 个里的 3 个上
  变成 **Pa 上最多 +47%** ⇒ **任何把 E 报到优于约 1.5 倍精度、或跨实例/跨场景比较 E 的主张都不被支持。**
  噪声底 ρ 5.7e-07 / E 2.2e-06 / ν 1.1e-07，表里每个差都高出 5–6 个数量级。
  ⚠️ **推翻 04 的预期**：这**不是**实例尺寸效应——猫 27k 高斯 → d_E 0.0006，
  而葡萄 13k → 0.351、娃娃 17k → 0.009；`n_gau`/IoU/池化权重占比**都排不出 d_E**。
  **是「哪些高斯不同」，不是「多少个不同」。**
  ⚠️ **HDBSCAN 在 bench 上是过切分而非过合并**：唯一的定性失败是一个 GT 物体被切成三块
  （IoU(A1, B6∪B7)=0.809），而合并侧的 `mu_spread` 反而**低于**单块侧
  ⇒ **没有证据表明它把不同材质并进了同一个实例。**

## Not yet specified

- **物性值的合理性本身**。06 号票的 sanity 旗标（**不是**本图的主张，本图不承诺物性质量）：
  step-10000 的头把墙/地板判成 E ≈ 2.3e10 Pa、**葡萄 1.4e10 Pa**、毛绒猫 2.5e6 Pa。
  **软猫合理，硬葡萄不合理。** 加上 06 量出的「E 在 SI 上有 ±47% 的成员集合敏感度」，
  真要对物性下任何量化断言之前，这条得先查——很可能就是
  [phys-on-query 14 号票](../phys-on-query/issues/14-checkpoint-diagnosis.md)
  那类「学生是不是在复读 `LUT[类别]`」的诊断，以**新头**为对象。

- **小/薄实例的物性不可信**。04 号票实测：2D↔3D 池化的 cos 在 >1000 高斯的实例上是
  0.95–0.99，在 534–1306 高斯的四个小/薄实例上塌到 **0.39–0.82**。
  这**给老图 11 号票（「3D 丢薄结构」）第一次提供了实测依据**——它当初是作为对 04 负反馈的
  随手补丁被判出 scope 的，现在有数字了，但仍不在本图内做（本图判据是「有个值就行」）。
  本图内的动作只有一条：**05 的实例物性表要把高斯数少的实例标出来**，不要让它们混在表里
  冒充等价可信的行。真要修是下一张图的事。

- **val 里的物性对照面板**（学生 / 常数 / 类别查表三行）。02 号票查明它不是「白送」：
  `compute_physics_metrics`（`src/evaluation/physics_metrics.py:80`）是按 SegVGGT 的 **query 口径**
  写的（吃 `query_masks [Q,S,h,w]` 做 IoU 最优匹配），`physgm_dpt` 没有 query；
  且自然插入点 `IGGTWrapper._log_physgm_predictions` 在 `if global_rank == 0:` 分支里
  （`iggt_wrapper.py:186-193`），从那里发 `sync_dist=True` **会挂死 DDP**。
  ⇒ 要接得先把 `physics_metrics` 从 query 口径解耦、并把调用提出 rank-0 块。
  本图不需要它（不承诺物性的量化声明），留给下一张图。

- **物性在 3D 空间的粒度**（逐高斯物性场 vs 实例级一个值）。01 号票的候选 C 会一并答掉它，
  但 C 已判出本图 scope ⇒ 这条 fog 留着，等本图收工后连同候选 C 一起重开。

- **论文主张**。用户明确：先看结果，发现什么对论文有利再反过来定。所以它不是一张票，
  是本图收工之后的动作。
- **物性质量的量化**。现在的立场是「有个值就行」。若导师看过图之后追问物性准不准，
  [phys-on-query 14 号票](../phys-on-query/issues/14-checkpoint-diagnosis.md)那类诊断
  （学生是不是 `LUT[类别]` 的复读）会以**新头为对象**回来，但现在连问题都不成形。
- **分割不够看时要不要训 IGGT 的 instance head**。现在明确不留口子（见"已定的事"第 1 条）。
  真到了非训不可那天，带实测数字重开——`docs/memory.md` 记着那套 mvc/disc loss 是民间实现、
  可能有误，重开时这是第一个要处理的东西。
- **多于 bench + garden 的场景**。alocasia / PhysicsDreamer 是唯一「本身就是物性研究对象」
  的高斯场景，但数据在失联挂载点上，重传还是重建（跑一遍 3DGS 训练）没估过成本。

## Out of scope

- **训 IGGT 的分割**（含 `instseg_iggt*` 那套民间 mvc_loss / disc loss 实现）。
  2026-09-09 已定：官方权重是本图唯一已知能看的东西，不动它。
- **SegVGGT 那条线的任何后续**。[segvggt-trace-3dgs 图](../segvggt-trace-3dgs/map.md)
  已标搁置。它的 10/11 号票（stuff/thing 界线、3D 丢薄结构）**判出 scope 并作废**：
  用户 2026-09-09 自陈那是对 04 负反馈的随手补丁、「没什么道理」，
  不该留在任何图上当阻塞项。
- **抬高物性上限**（重跑伪标签 / 换视角选择 / 改 prompt / 接外部材料数据库）：
  继承 [phys-on-query 图](../phys-on-query/map.md)，一定会做，不在本图。
- **类别预测**：继承 phys-on-query 图与老图的判定，本图只做 class-agnostic。
- **候选 B（训练时用预测 mask，聚类进训练循环）与候选 C（逐点物性场，改头 + 改 loss）**：
  [01 号票](issues/01-physics-from-dense-feat-to-3d-instances.md)判出本图 scope。
  不是「以后不做」——C 表示上更干净，还顺带答了「物性在 3D 空间的粒度」那条老 fog——
  而是**两者都在关键路径上加一个未知量，而候选 A 的代价是可量的**。重开是下一张图的事。

- **`physgm_copy` / `class` / `property` 三个 scheme**：G2 已判——前者是 scene-level 不满足
  「实例物性」，后两者的输出口径与 PhysGM 的归一化常数和 loss 对不上。
  本图只走 `physgm_dpt`（若 01 号票判它不成立，那时重开这一条）。

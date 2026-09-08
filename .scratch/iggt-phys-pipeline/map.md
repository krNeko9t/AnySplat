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

## Not yet specified

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

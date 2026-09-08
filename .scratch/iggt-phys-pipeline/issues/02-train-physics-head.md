# 02 — 物性头的训练接线与运行点，并起跑

Type: task
Status: closed
Blocked by: [01 — 物性怎么从 dense feat 走到 3D 实例](01-physics-from-dense-feat-to-3d-instances.md) ✅ closed
Blocks: [05 — 交付物](05-deliverables.md), [06 — 量两处不等价](06-quantify-the-two-mismatches.md)
Assignee: subagent (2026-09-09)

> **关键路径的长杆。** 地图 Notes：票半小时一张，**训练是唯一实打实的时间**
> ⇒ 这张票的目标是**今晚把训练起跑**，不是把它跑完。

## Question

把 `physgm_dpt_iggt.yaml` 从「scannet100 + 官方 IGGT」改成「Infinigen 物性语料 + 官方 IGGT」，
定死运行点，起跑，并留下一份能验收的 lock。

### 已知要改的（G6）

- **dataset 块整块换成 Infinigen**，直接抄 `phys_query_arm_b_lora.yaml:149-161`：
  root `/mnt/storage_pool/liaoyuanjun/data/InsScene-15K`、
  `manifest_infinigen_phys.jsonl`（1466 行）、
  `scene_split_path: config/experiment/splits/infinigen_phys_split.json`（1387 训 / 74 验）、
  `original_image_shape: [288, 512]` / `input_image_shape: [252, 448]`、
  `physics_parser: instascene_vlm_physgm`、`fixed_views_and_shape: true`、4 视角。
  ⚠️ 现在 `physgm_dpt_iggt.yaml` 里是 `processed_scannetpp_v2` + `[690, 920]`，**必须换**。
- **`hydra.run.dir` 挪出根盘** → `/mnt/storage_pool/liaoyuanjun/runs/`（地图 Notes：根盘 93% 满）。
- **`PHYSGM_NORMALIZATION` 不用动**（`parsers.py:74` 已经是本语料训练集拟合值）。

### 要定的

1. **step 数**。`physgm_dpt_iggt.yaml` 现在写 60000。frozen probe 只有 physics_scheme 有梯度，
   phys-on-query 的经验是**物性 5–10k 就触底后回退**（那是 SegVGGT 上的读数，仅作参照）。
   倾向：**max_steps 缩到 10000，val_check_interval 500**，两天里跑得完还留得下重跑一次的余量。
2. **lr**。现配 2.0e-4（抄 PhysGM 的 decoder 训练），`param_groups` 给 `physics_scheme`
   乘 1.0。frozen probe 下这个 lr 是唯一在动的东西，倾向照抄不动。
3. **冻结与 lock**。`freeze_keywords` 现为
   `["*aggregator*","*camera_head*","*point_head*","*depth_head*","*part_adaptor*","*part_head*"]`。
   ⚠️ **要现场核实这份清单是否真的把「除 physics_scheme 外的一切」都冻住了**
   —— IGGT 上还有 SamProjector / instance 相关模块；phys-on-query 09 号票做的
   glob + `!` 取反让 `["*", "!*physics_scheme*"]` 成为可写的、**表达上更严**的等价写法。
   起跑前生成 lock 并提交（`config/experiment/locks/`）。
4. **val 里要不要报物性对照**。`src/evaluation/physics_metrics.py` 已有学生/常数/类别查表三行，
   接上是白送的一个 sanity 面板（**不是**验收判据——本图不承诺物性的量化声明）。
   ⚠️ 若接：phys-on-query 13 号票的坑——`self.log` 要 `sync_dist=True`，
   且指标 dict 的 key 必须 `sorted()`，否则多卡会静默串号。

## 完成判据

训练**已经在跑**，且：lock 已生成并与实测指纹一致、可训参数只有 physics_scheme、
step 0 的 val 能读出数、输出目录在 `/mnt/storage_pool`。
把「预计跑完时间」写进 Answer，后面的票按它排。

---

## Answer（2026-09-09 00:36 关闭）

### 训练已经在跑，ETA 2026-09-09 ~02:04。

- **PID** 500516（DDP 子进程 501201/501202/501203），GPU 0/2/3/4，各约 10.9 GB、100% util
- **log** `/mnt/storage_pool/liaoyuanjun/runs/logs/physgm_dpt_iggt_2026-09-09_00-19-50.log`
- **run dir** `/mnt/storage_pool/liaoyuanjun/runs/exp_physgm_dpt_iggt/2026-09-09_00-19-57`
- **步速 0.614 s/step**（含 val + 3.2 GB checkpoint 写盘，每 500 步一次约 41 s）
  ⇒ 10000 步 ≈ **1.70 h**。干净窗口 0.535 s/step + 20×41 s 开销 = 1.71 h，两个算法一致。
- **盘**：run dir 在 step 1400 时 9.0 GB；ckpt 3.21 GB × (`save_top_k: 3` + `last`) ≈ 12.9 GB 封顶，
  `/mnt/storage_pool` 还有 500 GB。根盘没被碰。

### 配置改了四块（`config/experiment/physgm_dpt_iggt.yaml`）

1. **dataset 整块换成 Infinigen**（G6）：`InsScene-15K` / `manifest_infinigen_phys.jsonl` /
   `infinigen_phys_split.json` / `[288,512]`→`[252,448]` / `fixed_views_and_shape: true`。
   删掉 scannet100 的 root、manifest、`[690,920]`。
2. `hydra.run.dir` → `/mnt/storage_pool/liaoyuanjun/runs/...`
3. `trainer.max_steps` 60000 → **10000**（`val_check_interval: 500` 不动）
4. `freeze_keywords` → `["*", "!*physics_scheme*"]`

**没动**：`lr: 2.0e-4`、`param_groups`、`checkpointing`、头与 loss 的配置、`PHYSGM_NORMALIZATION`。

### 冻结验收

lock 重新生成：**1729 参数，134 可训（`T`），1595 冻结**。
`grep '^T ' | grep -v physics_scheme` → **0 行**。
134 个全是 `physics_scheme.physics_head.*`（DPT refinenet / window self+cross attn / 输出卷积）
与 `physics_scheme.physgm_dense_readout.decoders.{0,1,2}.*`（三个 per-property MLP）。
运行日志里三行对上：`'*' -> 1729 matched` / `'!*physics_scheme*' -> 134 matched` / `frozen 1595`。
`freeze_contract.verify()` 每个 rank 在 `setup("fit")` 里跑、不一致就抛，四个 rank 全过。

### 地图事实核验

- **G1 ✓** 权重路径本地且真实（run 已加载）
- **G6 ✓** manifest **1466 行**、split **1387 训 / 74 验**、
  `PHYSGM_NORMALIZATION`（`parsers.py:74`）逐位对上
  （density 2.863740/0.399147、E 9.495947/1.317972、ν 0.336525/0.066235），**没动**

### ⚠️ 一条要更正的记录：本票对旧 `freeze_keywords` 的怀疑是错的

本票第 3 条（和地图）假设旧的六前缀枚举
（`*aggregator*`/`*camera_head*`/`*point_head*`/`*depth_head*`/`*part_adaptor*`/`*part_head*`）
**可能漏掉 IGGT 的 SamProjector 和 instance 相关模块**。重生成 lock 后
**没有任何一行 `requires_grad` 变化**，`structure: sha256:43e9b3d4...` 也完全相同。

原因：这套 config 下的 IGGT 只建出七个带参数的顶层模块——
`aggregator`(1210 tensor) / `part_head`(119) / `part_adaptor`(73) / `camera_head`(69) /
`point_head`(62) / `depth_head`(62) / `physics_scheme`(134)。
**`physgm_dpt` 这条 build 里根本没有带参数的 SamProjector 或 instance head**，
所以旧枚举早就是精确等价的。

改仍然保留——`["*", "!*physics_scheme*"]` 对「以后加模块」是不变式，枚举不是——
但 yaml 注释已改成陈述实测事实，不再重复那条被证伪的说法。
**本次训练在新旧两种写法下逐位相同。**

### 没做的：val 物性对照面板（(g)），故意跳过，且背后是真风险不是时间

`compute_physics_metrics`（`src/evaluation/physics_metrics.py:80`）是为 SegVGGT 的 **query 分支**
写的：吃 `query_masks [Q,S,h,w]` logits，做 IoU 最优的 query↔GT 匹配。
IGGT 的 `physgm_dpt` 没有 query（直接用 GT `instance_mask` 池化）
⇒ 接上它意味着**造一个恒等匹配**，不是调用现成函数。

更要命的是自然插入点 `IGGTWrapper._log_physgm_predictions` 是在
**`if self.trainer.global_rank == 0:` 分支里**调用的（`iggt_wrapper.py:186-193`）。
本票自己的警告要求 `self.log(..., sync_dist=True)`——但从 rank-0 独占分支发起集合通信，
另外三个 rank 永远不进入，**会把 DDP 挂死**。要做对得先把物性指标调用提出 rank-0 块。
远超十分钟，且失败模式是训练静默卡死。面板本来就明确**不是**验收判据，
而 step-0 的 `[PhysGMPred]` JSON 块已经每次 val 给出逐实例 pred-vs-gt 读数
（如 density z=0.0244 → 747.3 kg/m³、E z=0.0287 → 3.42e9 Pa）。

⇒ **「接 val 物性面板」需要先把 `physics_metrics` 从 query 口径解耦、并把调用提出 rank-0 块**。
这条已记进地图 Not yet specified。

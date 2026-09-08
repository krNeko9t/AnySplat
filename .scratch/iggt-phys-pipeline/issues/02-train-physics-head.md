# 02 — 物性头的训练接线与运行点，并起跑

Type: task
Status: open
Blocked by: [01 — 物性怎么从 dense feat 走到 3D 实例](01-physics-from-dense-feat-to-3d-instances.md)
Blocks: [05 — 交付物](05-deliverables.md)
Assignee: —

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

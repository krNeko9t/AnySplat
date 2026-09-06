# 02 — 把 Infinigen 全量 VLM 伪标签接进训练

Type: task
Status: closed (2026-09-06)
Blocked by: —
Assignee: @krNeko9t (session 2026-09-06)

## Question

标签已经生成（Infinigen 完整数据集，naive 视角选择 + `pred_phys.prompt`），
但**不在本机**。要开训必须先把这条打通。

需要落定的事实（做完记在 `## Answer` 里，后面的票都依赖它们）：

1. 标签在哪（机器 / 路径 / 大小），怎么进到训练机。
2. 每条记录的实际 schema：至少要有 `(scene, instance_id, E, ν, ρ)`，以及能拿到物体类别的字段
   （Infinigen 的 Factory 类名）。类别字段是 04 的前提。
3. 单位与归一化：三个量各自的单位是什么，`log10 z-score` 的 mean/std **从哪个集合上算**
   （必须是训练集，且要存下来，否则推理期反归一化对不上）。
4. 覆盖率：多少实例有标签、多少缺失、缺失怎么处理（跳过 / 置 unknown / 不参与 loss）。
5. 与实例 GT 的对齐：伪标签的 `instance_id` 和分割 GT 的 id 是不是同一套。

**完成判据**：能在训练机上加载一批数据，打印出若干个实例的 (类名, E, ν, ρ) 且数值合理。

---

## Answer（2026-09-06 实做，全部在 GPU 服务器上跑通）

**一句话**：标签本来就在训练机上，缺的是 manifest。补了三个脚本、生成三份数据侧产物，
`DatasetManifest + instascene_vlm_physgm` 已能吐出 `(类名, E, ν, ρ)`。**02 不再阻塞任何票。**

### 0. 环境更正

`硬件环境.md` 里"开发机无 GPU"说的是另一台。**本 session 就在 GPU 服务器上**
（`bms-39468022-001`，8×A100-40G，conda env `anysplat`）。数据是本地盘，**不存在"传到训练机"这件事**。
纯 CPU 脚本要记得 `conda activate anysplat`，base env 没有 torch。

### 1. 标签在哪、多大

| | 路径（相对 `root=/mnt/storage_pool/liaoyuanjun/data/InsScene-15K`） | 大小 |
|---|---|---|
| 帧（RGB/Depth/ObjectSegmentation/Objects/camview） | `processed_infinigen/scene_XXX/<hash>/frames/camera_0/` | 303 G |
| VLM 伪标签 | `preprocessed/annotations/infinigen/infinigen/scene_XXX/<hash>/Qwen3.6-27B.json` | 205 M |

两棵树 **1466 : 1466 严格一一对应**，无孤儿。每场景 100 帧（3 个例外，见 §6），共 **146,034 帧**。
每场景另有 `Qwen3.6-27B.coverage.coverage.json`（生成期覆盖账本）与两份 shard 中间产物。

### 2. schema + 类别字段

记录字段恒为 `{id, instance_id, n_views, mode, response}`，53,328 条，`id` 无重复，
`mode` 恒为 `compose`，`n_views` 分布 4/1/2/3 = 45530/3722/2294/1782。
`response` 里除 `physical_property` 外还白送 `object_description`、`scene_role`、
`geometry_form`、`attachment_relation`、`appearance_materials`（含 prototype+score）、
`physical_priors`（heaviness/stiffness/friction 的 bin + confidence）——**目前一个都没用**。

`physical_property` 三个键齐全、0 条畸形：
`"Density (kg/m³)"` / `"Young's modulus (MPa)"` / `"Poisson's ratio"`，各带 `{mean, variance}`。

**类别字段：标签里没有，但可无损恢复。** `frames/Objects/camera_0/Objects_*.json` 把
`object_index → blender 物体名` 映死，而 `object_index` **正是** `ObjectSegmentation` 里的像素值、
也正是标签记录的 `id`。名字形如 `BedFactory(1961782).spawn_asset(7702850)` ⇒ Factory 类名 `Bed`；
房间构件形如 `bedroom_0/0.wall` ⇒ 记为 `room:wall`。

已落盘：`infinigen_class_table.json`（4.2 M，`{scene_id: {inst_id: {name, class}}}`）。
**51,981 / 53,328 = 97.47% 命中，213 个 distinct 类。**
Top：Window 5717、CeilingLight 4432、room:wall 3361、room:floor 2648、BookStack 2168、room:ceiling 1950。

未命中的 1,347 条**全部是 `id == 0`**（1466 个场景里 1347 个各有一条）。抽样看描述，
它们是**窗玻璃 / 镜面**这类 Infinigen 没分配 `object_index` 的东西，在 mask 里就是背景，
**本来也无从匹配**。parser 的 `_iter_vlm_instances` 已经 `inst_id <= 0` 直接丢弃 ⇒ 行为一致，不用改。

### 3. 单位与归一化 —— **这里有个坑**

单位：密度 kg/m³、杨氏模量 **MPa**、泊松比无量纲。
`InstasceneVlmPhysGMParser` 的变换是 `z = (log10(max(v·si_scale, 1.0)) - mean) / std`，
E 的 `si_scale=1e6`（MPa→Pa），密度/泊松比 `si_scale=1`，泊松比不取 log。

**`PHYSGM_NORMALIZATION`（`src/dataset/physics/parsers.py:52-57`）的 mean/std 是从 PhysGM 仓库
逐字抄来的常数，不是本语料拟合的。** 差得不小（`scripts/fit_physgm_norm.py` 全 1466 场景实测）：

| 量 | 拟合 mean | 拟合 std | 在用 mean | 在用 std | ⇒ 实际 z 的 (均值, 标准差) |
|---|---|---|---|---|---|
| density (log10 kg/m³) | 2.8656 | 0.3993 | 3.0 | 0.5 | (−0.27, 0.80) |
| youngs_modulus (log10 Pa) | 9.4983 | 1.3206 | 7.3872 | 2.4565 | **(+0.86, 0.54)** |
| poisson_ratio (raw) | 0.3363 | 0.0663 | 0.398 | 0.111 | (−0.56, 0.60) |

E 尤其离谱：目标整体偏 +0.86、被压到 0.54 倍宽。**这直接改变 Gaussian NLL 里 μ 与 σ 的相对尺度**，
不是无害的仿射。**换不换是训练配方决策，已开 [07 号票](07-physgm-norm-constants.md)**（阻塞于 03，
因为 train-only 拟合要等划分定下来）。已落盘 `physgm_norm_infinigen.json`（全语料拟合，report-only）。

非正数：density 27 条、E 25 条 mean ≤ 0（占 0.05%），被 `max(v, 1.0)` 夹到 log10=0，不崩。泊松比 0 条。

### 4. 覆盖率

- **生成期账本**（1466 份 coverage 全 `complete: true`）：eligible 53,328，completed 53,328，
  **failed 0，missing 0**，另有 **skipped_no_valid_view 25,594**（候选物体的 32.4% 因为没有可用视角而根本没送 VLM）。
- **相对可见实例**（抽 25 场景 × 10 帧实测）：mask 里出现过的非零 id 中 **75.3% 有标签**，
  即 **24.7% 的可见实例无物性监督**；另有 5.6% 的已标注 id 在抽到的帧里没出现。
  像素口径（含 id=0 背景）覆盖 95.4%。
- **缺失怎么处理：已经对了，不用动。** parser 建 `valid[max_id+1]` 布尔 LUT，只写有标签的 id；
  `loss_physgm.py:67` 先 `in_range = ids_b < valid.shape[0]` 再 `valid[ids_ok]` 过滤
  ⇒ 越界 id 和无标签 id 都不进 loss，不是"置 0 当监督"。

### 5. 与实例 GT 的对齐 —— **同一套 id，无需映射**

`ObjectSegmentation_*.npy`（int64，288×512）的像素值 == `Objects_*.json` 的 `object_index`
== 标签记录的 `id`。实测某帧 43 个 unique 值里 42 个能在 Objects json 里查到名字，
唯一查不到的就是 0。

### 6. 落地产物

新增脚本（都在 `scripts/`，纯 CPU）：

- `make_manifest_infinigen_phys.py` — 走 `processed_infinigen`，复用既有
  `make_manifest_infinigen.make_scene_manifest` 造 frames，把路径改写成 root-relative，
  挂上 `physics_labels_path`。scene_id = `infinigen_<scene_XXX>_<hash>`。
- `build_infinigen_class_table.py` — 产出 §2 的类别表。
- `fit_physgm_norm.py` — 用与 parser **完全相同**的变换拟合 mean/std；`--scene_ids` 传训练集划分。
- `check_infinigen_phys_batch.py` — 完成判据的验收脚本。

新增数据（在 dataset root 下，未进 git）：
`manifest_infinigen_phys.jsonl`（101 M / 1466 行）、`infinigen_class_table.json`、`physgm_norm_infinigen.json`。

复现：
```bash
conda activate anysplat
R=/mnt/storage_pool/liaoyuanjun/data/InsScene-15K
python scripts/make_manifest_infinigen_phys.py --root $R --out $R/manifest_infinigen_phys.jsonl
python scripts/build_infinigen_class_table.py  --root $R --out $R/infinigen_class_table.json
python scripts/fit_physgm_norm.py --root $R --manifest $R/manifest_infinigen_phys.jsonl --out $R/physgm_norm_infinigen.json
python scripts/check_infinigen_phys_batch.py --root $R --manifest manifest_infinigen_phys.jsonl \
  --class_table $R/infinigen_class_table.json
```

要开训，`config/dataset/manifest.yaml` 需要：`root` 指到 dataset root、
`manifest_path: manifest_infinigen_phys.jsonl`、`physics_parser: instascene_vlm_physgm`、
`original_image_shape: [288, 512]`。**具体写进哪份 experiment config 归 [05 号票](05-training-operating-point.md)。**

### 7. 完成判据：达成

`check_infinigen_phys_batch.py` 用训练同款 `bounded_fixed` sampler 真的取了样本
（`image=(4,3,224,392)`，`instance_mask=(4,224,392)`），实打实的输出：

```
=== infinigen_scene_000_16255241  labelled=48  其中 35 个 id 出现在抽到的 4 帧 mask 里 ===
   39 Bed            E=1.5 MPa      nu=0.450  rho=45
   42 Mattress       E=1.5e4 MPa    nu=0.350  rho=700
   43 Pillow         E=0.05 MPa     nu=0.490  rho=40
   45 Towel          E=1.1e4 MPa    nu=0.350  rho=650
   49 Bed            E=8000 MPa     nu=0.350  rho=350
```

⚠️ `DatasetManifest` 会按 `num_context_views` 丢掉视角不够的场景：
`infinigen_scene_163_42a2ab3f`(2)、`infinigen_scene_165_1eac55ca`(1)、`infinigen_scene_169_a8d577e`(1)。
**1466 → 1463**。

### 8. 顺手看到的、留给别的票的东西

- **类内方差肉眼可见地不为零**（04 要量的正是它）：同场景两张 `Bed` 一个 E=1.5 MPa/ρ=45（泡沫），
  一个 E=8000 MPa/ρ=350（木头）；`Pillow` 跨 0.05/1/5 MPa；`room:wall` 跨 5 MPa/ρ=150 与 1e4 MPa/ρ=650。
  ⇒ **`H(P_VLM | 类别) ≈ 0` 这个最坏情形基本可以排除**，但 04 仍要出数。
- **但方差里混着"认错东西"**：`Towel` 拿到 E=1.1e4 MPa / ρ=650 是木头的数。
  这是"教师真在看图"还是"教师把 mask 对应错了物体"，两者对结论意义完全相反。**已写进 04 的正文。**
- **`room:*` + `Window` 占 28.6% 的标签**，这些东西不适合物理模拟。已开
  [08 号票](08-non-object-classes.md)（阻塞于 04），并把"数据清洗"那块迷雾正式毕业过去。
- `n_views` 有 3722 条只用了 1 个视角、2294 条 2 个 —— 这批标签的置信度理应更低，
  但目前 loss 对它们一视同仁。`response` 里的 `variance` / `confidence` 也全丢了。**暂不开票。**

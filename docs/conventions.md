# AnySplat Conventions

本文记录仓库内和实例 ID、聚类 label、mask、depth 相关的隐性约定，避免不同模块之间重复猜测 `0`、`-1`、`+1` 的含义。

## Instance ID 主约定

GT、loss、可视化、trace 后处理的主约定是：

- `0` 表示 `background / ignore / invalid`，不作为有效实例参与训练、统计或导出。
- 有效实例 ID 从非零开始，通常是 `1..K`。
- GT instance mask 可以是不连续 ID；进入可视化时通常会重排为连续的 `1..N`，并保留 `0` 给背景。
- 二值 mask 中 `0` 表示背景或无效区域，非零表示前景或有效区域。

对应代码里，`ignore_id` 默认应保持为 `0`。如果读取外部数据源时发现 `0` 不是背景，必须在数据边界显式转换并写注释。

## 聚类 Label 约定

HDBSCAN / sklearn / k-means 的原始聚类语义和仓库输出语义不同：

- 原始 HDBSCAN / DBSCAN：`0` 是第一个有效簇，`-1` 是噪声或未聚类。
- 原始 k-means：`0` 是第一个有效簇，没有背景槽位。
- 仓库对外保存或可视化聚类结果时，默认要把有效簇整体 `+1`，输出为 `0 = background / invalid / unassigned`，`1..K = cluster IDs`。

因此，聚类函数如果会被下游当作实例/可视化 label 使用，应尽量在函数边界完成 `+1`，不要要求每个调用点手动补偏移。

当前约定边界：

- `src/instseg/hdbscan_assign.py`：内部遵守 HDBSCAN 语义，返回 `0..K-1` 的连续簇 ID；噪声会通过 NearestCentroid 被补分配，不再保留 `-1`。
- `iggt_idmap.py::cluster_features_hdbscan()`：对 `hdbscan_assign()` 的返回值做 `+1`，对外输出 `0 + 1..K` 语义。
- `src/visualization/instance_viz.py::cluster_instance_embeddings()`：k-means 对外返回 `0 = invalid/background`，有效簇为 `1..K`。
- `scripts/instseg_infer.py` 与 `scripts/trace_instance_to_gaussians.py` 的 3D 聚类输出均应保持 `0 + 1..K` 语义。

## Depth 和 Valid Mask

Depth 相关约定是：

- depth `0` 表示无效深度。
- 有效深度通常由 `depth > 0`，`isfinite(depth)`，以及小于数据集无效 sentinel 的条件共同决定。
- 当 `valid_mask` 与 `instance_mask` 合并时，无效深度区域的 instance ID 会被置为 `0`，也就是 loss 里的 ignore/background。

这意味着如果某个数据源在无效 depth 区域仍有非零 instance ID，这些像素会被训练侧静默忽略。

## Physics Label 约定

Physics 监督不是 instance ID，不能混用两套编号。仓库有三套平行方案（Hydra `phys_scheme` 切换）：

### scheme `class`（`PhysicsTarget`）

- instance ID：`0 = ignore/background`，非零为实例。
- physics class：`0 = ignore / unlabeled`，有效类别从 `1` 开始（LUT 存 1-indexed）。
- 进入 cross entropy 前，`resolve_instance_ce` 把 `1..C` 转成 `0..C-1`。
- 类别名与字符串→id 映射只活在 `src/dataset/physics/parsers.py`（如 `3dovs_json`），经 `PhysicsTarget.class_names` / `label_lut` 传出；不要在 dataset / wrapper / loss 里再写一份映射。

### scheme `property`（`PhysicsPropertyTarget`）

- 属性顺序权威：`PROPERTY_NAMES = (density, youngs_modulus, poisson_ratio)`（`types.py`）。
- JSON key → 列映射只活在 `parsers.py`（`instascene_vlm`）；下游禁止再解析 raw key。
- `valid[instance_id] == False` ⇒ ignore。**`id=0` 永远 ignore**（ScanNet VLM 标注里 id=0 不是真实物体，与 instance `ignore_id=0` 一致）。
- mask 中出现但标注缺失 / `physical_property` 不完整的实例 → `valid=False`。
- 监督空间由 parser 一次变换（loss 只做公式）：
  - density / youngs_modulus：`mean_lut = log(raw_mean)`，`log_var_lut = log(raw_var + eps)`
  - poisson_ratio：`mean_lut = raw_mean`，`log_var_lut = log(raw_var + eps)`

### scheme `physgm_copy`（`PhysGMTarget`）

- 属性顺序 / `id=0` ignore / `valid` 语义与 scheme `property` 相同。
- 监督空间照抄 PhysGM：z-score 后的标量（`value_lut`），无 GT 方差 —
  预测方差由 loss 的 Gaussian NLL 学出（aleatoric，不回归 VLM 的 variance）。
  - density：`(log10(kg/m³) - 3.0) / 0.5`
  - youngs_modulus：`(log10(Pa) - 7.387210) / 2.456477`（PhysGM `E_MEAN/E_STD`；VLM 的 MPa 先 ×1e6）
  - poisson_ratio：`(raw - 0.398) / 0.111`（PhysGM `NU_MEAN/NU_STD`）
- 归一化常量与反归一化 helper 只活在 `parsers.py`（`PHYSGM_NORMALIZATION` / `physgm_denormalize`）。

## 可视化和导出约定

可视化默认保留 `0` 给黑色背景：

- `colorize_labels(..., ignore_label=0)` 会让 label `0` 保持黑色。
- GT 可视化会丢弃原始 `0`，再把其他 ID 重排成 `1..N`。
- 聚类可视化需要输入 `0 + 1..K` 语义，否则第一个簇可能被画成背景。

导出默认跳过背景：

- per-cluster PLY 导出使用 label `1..K`，`0` 是未聚类或背景。
- GT instance split 默认 `ignore_id=0`，不导出 `instance_00000.ply`。

## 数据源边界

数据加载和 manifest 生成脚本通常只记录路径、原样读取 ID，不负责猜测数据源语义：

- Infinigen / ScanNet++ / RE10K 等数据若约定 `0` 为无效或跳过，进入仓库后会自然符合主约定。
- manifest 生成脚本不应偷偷重映射 ID；如果需要重映射，应在脚本和输出说明里显式写清楚。
- `iggt_idmap.py` 生成的 id map 已经按仓库主约定保留 `0`，有效 cluster 从 `1` 开始。

## 修改相关代码时的检查清单

- 新增聚类输出时，确认对外是不是 `0 + 1..K`。
- 新增可视化时，确认 `0` 是否应保持黑色。
- 新增 loss 时，确认 `ignore_id=0` 是否被排除。
- 新增数据集时，确认外部 mask 的 `0` 是否确实是 background / ignore / invalid。
- 新增 trace 或导出功能时，确认是否跳过 `ignore_id=0`。

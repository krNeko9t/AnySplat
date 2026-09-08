# 03 — 官方 IGGT 权重的实例基线 + HDBSCAN 代价

Type: task
Status: open
Blocked by: —
Blocks: [05 — 交付物](05-deliverables.md)
Assignee: —

> **不阻塞于任何票，安排在 02 的训练跑着的时候并行做。**
> 它同时产出两样东西：本图的**验收对照组**，和用户点名要重新评估的 **HDBSCAN 代价**。

## Question

在 `3dovs/bench` 和 `mipnerf360/garden` 上，用**官方 IGGT 权重**
（`/home/liaoyuanjun/iggt_checkpoint.pth`）跑完整的 trace → HDBSCAN → 3D 实例，
把「能看」这件事从印象变成一份可以并排比的产物。

### 要产出的

1. **验收对照组**：bench 上的**全分辨率实例叠加图**（原图 + 半透明实例色）。
   ⚠️ 老图 04 的教训：**不要**用 1/4 分辨率灰底 3DGS 渲染去验收，上次就是这么被判不合格的，
   而同一份结果叠回原图边界是贴着物体走的。
   本图 Destination 的判据（"并排不明显更差"）里的**那个"排"就是这张图**。
2. **HDBSCAN 的代价读数**（用户点名）：`src/instseg/hdbscan_assign.py` 是
   subsample → HDBSCAN → NearestCentroid 全量赋值。要量：
   - bench（约 10⁶ 高斯）和 garden 上的**墙钟**与**峰值内存**；
   - `max_points` 的 subsample 上限**够不够**——subsample 到多少之后实例数/边界开始变；
   - 参数敏感度：`min_cluster_size` / `min_samples` / `cluster_selection_epsilon`
     （`instseg_infer.py:88-90` 的默认是 50 / 10 / 0.06）动一动，实例数变多少。
     **一个只在 bench 上成立的参数不算数**，garden 上要用同一组。
3. **实例数与形态**：bench 出几个实例、有没有"不是任何东西"的碎片、
   墙和地板（bench GT 里 62% 的像素是 stuff）被聚成了什么。
   **只记录，不修**——本图不设 stuff/thing 规则（老图 10 号票已判出 scope 并作废）。

### 顺手要确认的两件事

- **场景路径**。⚠️ 老图记着 `configs/trace/*.json` 里的场景路径**全部指向失联的
  `/mnt/shared-storage-gpfs2`**。本机实有的两组场景在
  `/home/liaoyuanjun/projects/instascene_preprocessed_data{,_zxl}/3dovs/bench`。
  先核实 garden 在不在本机，**不在就当场说，别硬跑**。
- **相机对齐**。老图 05 号票已证 bench 上整图 resize 对 `TraceCamera` 是无操作
  （`trace_cams = list(cam_list)`，`trace_crop_aligned=False`）。
  garden 若是同尺寸横图，同一结论直接继承；**不同尺寸就要停下来说**。

## 完成判据

一张 bench 的全分辨率叠加图（作为后续所有验收的对照组）+ 一份 HDBSCAN 代价与参数敏感度的读数
+ garden 上同一套脚本跑通与否的结论。**不要求改任何东西。**

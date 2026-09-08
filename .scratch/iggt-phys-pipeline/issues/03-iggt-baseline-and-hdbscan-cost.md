# 03 — 官方 IGGT 权重的实例基线 + HDBSCAN 代价

Type: task
Status: closed
Blocked by: —
Blocks: [05 — 交付物](05-deliverables.md)
Assignee: subagent (2026-09-09)

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

---

## Answer（2026-09-09 01:15 关闭）

### 验收对照组（交付物 1）——成了，是本图之后一切并排的基准

`/mnt/storage_pool/liaoyuanjun/runs/ticket03/overlay_bench_default/contact_sheet.png`
6 帧 bench，**每帧 1008×756 原生分辨率**，原图在下、半透明实例色 + 2px 描边在上，底部图例带像素占比。
仓库内 JPEG 副本：`.scratch/iggt-phys-pipeline/figures/03_*.jpg`（7 张）。

**没有任何灰底**：合成的 alpha 由光栅器累积的 splat 覆盖驱动，未建模的像素保留照片。
bench 6 个视角覆盖率都是画面的 100%。
**我（orchestrator）逐张看过**：doll / cat / 蛋挞 / 玩具车 / 葡萄 / 叶子边界都贴着物体走，
墙和地板各是一整块干净实例。**六个视角里每个物体保持同色**
⇒ 这张图**顺带把 05 号票的交付物 4（新视角一致性证据）也兑现了**，而且是在叠加图上而非灰底渲染上。

### HDBSCAN 代价（交付物 2）

基线 = `instseg_infer.py` 默认 50 / 10 / 0.06，`max_points` 200000，`knn_k=20` 平滑之后。

| 阶段 | bench（1.006 M 有效） | garden（1.753 M 有效） |
|---|---|---|
| subsample | 0.024 s | 0.024 s |
| **HDBSCAN** | **9.33 s** | **10.88 s** |
| NearestCentroid（全量） | 0.18 s | 0.27 s |
| 连续重标号 | 0.11 s | 0.19 s |
| **合计** | **9.64 s** | **11.36 s** |
| 峰值 RSS / 增量 | 1.01 GB / +166 MB | 1.10 GB / +212 MB |
| 实例数 | 9 | 82 |

**代价由 cap 决定，不由场景大小决定**：garden 多 74% 的高斯，只贵 18%。
只有 NearestCentroid 随 N 增长，而它占 2%。峰值 RSS 恒在 ~1.1 GB。**这不是内存问题。**

⚠️ **真正的大头不在 `hdbscan_assign` 里**：`postprocess_knn_k=20` 的 cKDTree 平滑
bench **13.87 s / 1.78 GB**、garden **16.98 s / 2.70 GB**，**比 HDBSCAN 本身还贵**。
trace 可忽略：bench 41.4 it/s（36 视角 0.87 s）、garden 43.5 it/s（185 视角 4.3 s）。

### `max_points` 够不够——曲线（锚在出厂的 200000）

| max_points | bench K | ARI / 变动 | bench HDBSCAN | garden K |
|---|---|---|---|---|
| 1000000 | 8 | 0.419 / 22.2% | 74.7 s | 40 |
| 500000 | 9 | 0.239 / 39.0% | 27.3 s | 56 |
| **200000** | **9** | 1.000 / 0% | **8.8 s** | **82** |
| 100000 | 10 | 0.928 / 3.3% | 3.7 s | 85 |
| 50000 | 11 | 0.947 / 3.4% | 1.6 s | 67 |
| 20000 | 8 | 0.947 / 1.6% | 0.53 s | 38 |
| 5000 | 6 | 0.947 / 3.0% | 0.13 s | 16 |

**结论：cap 绰绰有余，而且调高会变差。** 两件只看「够不够」会漏掉的事：

1. **`max_points → N` 不收敛。** 它不是近似旋钮——HDBSCAN 的密度估计是采样密度的函数，
   改它就是改目标函数。bench 在 500000 上**墙和地板并成一个 89.1% 的簇**
   （`overlay_bench_maxpoints_grid.png` 第二行可见），1000000 上地板的颜色成片漏到墙上。
   两者都比 200000 贵 3–8×**且更难看**。
2. **两个场景对「能降到多低」意见不一致**：bench 能忍到 20000（1.6% 标签变动，快 16×），
   garden 不能（K 从 82 塌到 38）。跨场景安全下限 ~100000。**保持 200000。**

交叉列联显示不稳定集中在 stuff 簇上：只看物体（非 stuff）高斯，
与 1 M 参照的一致率在整个范围里稳定在 80–86%。**物体是稳的，墙/地板那一刀是双稳态的。**

### 参数敏感度——带上跨场景约束

`cluster_selection_epsilon`（主导旋钮，也是唯一会破坏跨场景的）：

| eps | bench K / 最大簇 | garden K / 最大簇 |
|---|---|---|
| 0.00 | 128 / 53.7% | 345 / 9.4% |
| 0.04 | 18 / 53.7% | 166 / 23.2% |
| **0.06** | **9 / 55.3%** | **82 / 24.7%** |
| 0.10 | 5 / **91.6%** | 16 / **65.8%** |
| 0.25 | 2 / 93.4% | 2 / 94.9% |

**0.06 是网格里唯一在两个场景上都不退化的值**——≥0.10 两边都塌（bench 并墙+地板，
garden 并成一个 66% 的团），≤0.04 两边都过碎。
**出厂默认扛住了跨场景检验，这是交付物 2 的承重结论。**

`min_cluster_size`（ms=10）：bench 50/100/200 标签完全相同（ARI 1.000），400+ 塌成单个 91.5% 簇；
garden 50→82、100→60、200→31、400→19。**两场景安全区 50–200。**
`min_samples`（mcs=50）：bench 几乎不敏感（1–200 内 ARI ≥ 0.87）；garden 敏感
（1/10/50/200 → K 49/82/122/62，子采样噪声 2.3%→32.2%）。**保持 10。**

⇒ **建议：什么都不改。** 50 / 10 / 0.06 @ `max_points=200000` 是网格里两场景上最好的点。

### bench 的实例数与形态（交付物 3）

**9 个实例**，1,005,708 / 1,045,236 被 trace（96.2%）。

| # | 是什么 | 高斯数 | 平均像素占比 |
|---|---|---|---|
| 8 | **墙**（stuff） | 555,810 | 42.6% |
| 9 | **地板/甲板**（stuff） | 366,893 | 40.9% |
| 1 | 猫摆件 | 27,402 | 5.1% |
| 3 | 娃娃 | 18,876 | 4.0% |
| 6 | 葡萄 | 13,434 | 2.4% |
| 7 | 葡萄叶 | 7,840 | 1.9% |
| 4 | 玩具车 | 7,581 | 1.6% |
| 2 | 蛋挞 | 6,847 | 1.3% |
| 5 | **叶/葡萄接缝上的碎片——不是任何东西** | 1,025 | 0.2% |

**2 个 stuff + 6 个真物体 + 1 个碎片。** 墙和地板各自成一整块，既没碎也没并。
除 #5 外没有簇小于全场景的 0.1%。
与 `sam/mask` GT 比：IGGT 把猫保持完整而 SAM 把猫头切开；IGGT 把叶子从葡萄里分出来而 SAM 并着。
**两边都说不上谁明显错。只记录，不修**（本图不设 stuff/thing 规则）。

`configs/trace/bench.json` 自带的 700/50/0.06 给 **7 个实例**：同样的墙/地板切分、
同样 6 个物体，但叶子并进葡萄、碎片 #5 消失。两套参数都站得住，**我没改 config。**

### garden 判定（交付物：只证不崩）

**garden 在本机，脚本能跑，但本机唯一的 garden 3DGS PLY 是一个坏文件。**

- **图像 ✓**：185 帧在
  `/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/garden/images`，
  **全部 5187×3361、横图、fx/fy 一致** ⇒ 老图 05 号票的相机结论**原样继承**
  （`trace_cams = list(cam_list)`、`trace_crop_aligned=False`），不需要停下来。
- **相机**：garden **没有 COLMAP `sparse/`、也没有 transforms JSON**，而 `CAMERA_BACKENDS`
  只吃 `colmap` / `nerf_transforms` ⇒ 脚本按出厂状态**加载不了 garden**。
  （同目录的 `bicycle`、`room` 都有 `sparse/`，唯独 garden 没有。）
  为此写了 `scripts/make_transforms_from_3dgs_cameras.py` 从 3DGS 的 `cameras.json` 转换，
  并**用渲染验证了约定**（`garden_camcheck/opencv.png`：桌、花瓶、球、地砖与照片对齐；
  `nerfstudio` 轴向明显错，平均绝对误差 30 vs 73）。
- ⚠️ **`.../official/mipnerf360/garden/point_cloud.ply` 是截断的。**
  header 声明 5,834,784 个顶点，文件里只有 1,839,236 行完整数据——**一次中断拷贝的 31.5%**。
  （我复核：文件大小 456,132,096 B = **恰好 435 MiB 整**，典型的中断拷贝；
  修改时间 Sep 4 20:27，而同目录 `bicycle`/`room` 都是 Jul 15 且大小正常。）
  **它是 `/mnt/storage_pool` 与 `/home/liaoyuanjun` 下唯一的 garden PLY**（我 find 过，确认）。

为回答「崩不崩」，把 header 改成实际行数，在这个**31.5% 的残缺场景**上跑了全流程
（明确标注，不冒充完整 garden）：185 视角、1,839,236 高斯、95.3% 被 trace、trace 4.3 s、
82 个实例、**不崩**。叠加图上桌/花瓶/草坪/地砖/桌腿是连贯实例且跨越 185 帧保持同一 id；
植被碎成几十个 blob。

⇒ **05 号票该承诺的**：garden 的**相机、图像、trace、聚类、叠加全链路已跑通并验证**，
**唯独缺一份完整的 PLY**（约 1.45 GB，需要重拷）。

### ⚠️ 五条与地图/票面冲突的实测，逐条报上来

1. **bench GT 是 84.2% stuff 像素，不是 62%。** 36 帧 `sam/mask` 上，
   每帧最大的两个 mask 平均占 84.2%（min 81.5%、max 85.8%）。**stuff 问题比地图假设的更重。**
2. ⚠️ **`sam/mask` 的 id 不是跨视角一致的。** 帧 00 是 id 1–8、帧 12 是 1–7、帧 24 是 1–9，
   **同一个 id 在不同帧指不同物体**（`photo_gt_iggt_sheet.png` 可见）。它是逐帧 SAM 输出。
   ⇒ **任何把它当 3D 实例 GT 的指标都是错的**，只有「逐视角 IoU + 匹配」是正当用法。
   老图记的「8 个实例」是**帧 00 的计数**，不是场景计数。
3. **场景路径那条说法被夸大了。** 票面说 `configs/trace/*.json` 的场景路径「全部指向失联挂载点」——
   不是：`bench.json`、`bench_gt_idmap.json`、`bench_precomputed_depth.json`、
   `bench_anysplat_instseg.json`、`alameda.json` 用的都是**相对** `source_path`，
   只有 `alocasia.json` 的场景路径在 `/mnt/shared-storage-gpfs2` 上。
   **真正指着失联挂载点的是 6 个 config 里 3 个的 `iggt_model_path`**——一行就能改，因为 G1 的本地权重路径是对的。
4. ⚠️ **地图上没有的新风险：IGGT 的实例特征空间是 batch-relative 的。**
   同样 4 个视角，单独跑 vs 放在 8 张的 batch 里 vs 放在 48 张的 batch 里，
   逐像素实例特征会变：平均 cosine 0.958 / 0.947，**5 分位 0.76，最小 0.19**。
   控制组：同一分组跑两遍 cosine = 1.0000 精确 ⇒ 编码器是确定性的，**漂移全部来自 batch 组成**。
   bench 碰不到（36 视角一次前馈）。**garden 的 185 视角被切成 4 个 48 的 batch，
   于是全场景 HDBSCAN 混了 4 个朝向不同的特征空间**——而漂移量（≈0.05）
   与 `cluster_selection_epsilon`（0.06）**同量级**。
   实测 garden 的主要簇扛住了，但**任何视角数多于一次前馈的场景都踩着这个雷**。
5. **F4 的噪声底传不到标签上。** 端到端控制组：两次独立 bench trace 在归一化特征上差
   maxabs 7.45e-8（相对 4.14e-7，与 F4 在裸 `gau_sem` 上的 1.7e-6 自洽），`num_ray` 逐位相同，
   **得到的 9 个实例完全相同（ARI 1.0000，0.00% 变动）**。
   `hdbscan_assign` 在固定输入上重复也逐位相同。
   ⇒ **上面每个数字的噪声底是「零标签变动」**，所以每一处差异都清得过噪声底。

### 新增文件（没碰任何共享文件）

`scripts/hdbscan_cost_sweep.py`、`scripts/render_instance_overlay.py`、
`scripts/make_transforms_from_3dgs_cameras.py`、`.scratch/iggt-phys-pipeline/figures/03_*.jpg`。

### 没做

- **没有真 garden 的数字**（上面 garden 的所有读数都来自 31.5% 残缺 PLY，
  作为「跑不跑得动」和一个 1.75 M 高斯的代价数据点有效，完整 garden 的实例数会不同）。
- **没测 GPU 峰值显存**（票面问的是 HDBSCAN 的峰值内存，那是 CPU 侧，量的是 RSS）。
- **没有拿 `bicycle` 顶替 garden**（它完好、6.13 M 高斯、原生 COLMAP，
  但它不是票面点名的场景，不想把 garden 的判定搅浑）。→ **已转交 05 号票执行。**

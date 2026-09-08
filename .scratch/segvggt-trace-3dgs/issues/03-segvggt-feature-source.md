# 03 — 把 SegVGGT 接成 trace 脚本的一个 feature source

Type: task
Status: resolved
Blocked by: [02 — 让 trace 支持 feat_dim > 20](02-chunked-trace-over-20-channels.md)
  （[05 — 预处理与 trace 相机对不对得上](05-preprocess-camera-alignment.md) 已关闭 2026-09-08）
Blocks: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Assignee: krNeko9t

## Question

`scripts/trace_instance_to_gaussians.py` 已有四个 feature source
（`anysplat` / `iggt` / `precomputed` / `gt_idmap`，见 `FEATURE_PREP`）。
本票加第五个：**`segvggt`**。

它比前四个多吐两样东西——现有的 `prep` 契约只返回
`{feat_maps, trace_cams, feat_dim, masks?, id_codec?}`，装不下：

| 要带出来的 | shape | 用途 |
|---|---|---|
| `feat_maps` | `[V, 128, H, W]` | trace 的输入（现有契约里已有） |
| `query_embed_proj` | `[N_b, 128]` 每批 | 04 号票并成 `Q_all`，与场做点积 |
| `query_phys_mu/var` | `[N_b, 3]` 每批 | 实例物性，按 `query_idx` 取行 |
| `scores` | `[N_b]` 每批 | 3D 去重时的优先级 |

⇒ **`prep` 的返回契约要扩一个可选字段**（例如 `query_bank`），
而不是把 query 硬塞进 `feat_maps` 或另开一条平行的主流程。

## 实现要点

- ckpt：`/mnt/storage_pool/liaoyuanjun/runs/exp_phys_query_arm_b_lora/2026-09-07_17-09-03/checkpoints/epoch_114-step_20000.ckpt`。
  加载走 `SegVGGTModel.from_checkpoint`（`src/model/arch/segvggt.py:350`），
  注意 `_assert_query_physgm_coverage`（`:297`）会校验 `query_physgm` 权重全部落位。
- **分批**：`encoder_batch_size` 复用现有参数，但对 SegVGGT 它的含义是"一次 forward 几个视角"，
  且训练分布是 4 —— 默认值要按 05 号票的结论定，不要沿用 iggt 那个 48。
- **解码复用现成的**：`scripts/segvggt_infer.py:85` 的 `decode_instances`
  和 `:137` 的 `report_physics` 已经写好了整套（含 `class_agnostic` 分支、
  `query_idx` 连接键、`physgm_denormalize` 到 SI）。**抽出来共用，不要重写一遍**，
  否则两处阈值会漂。
- `class_agnostic=True` 写死（地图 F3：这个 ckpt 不预测类别）。
- 预处理与相机映射按 05 号票的结论落成代码。

## 判据

- 在 `3dovs/bench` 上跑完不崩，落盘 `gau_feat [P,128]`、`num_gsem [P]`、
  以及各批的 `Q_all` / 物性 / score。
- `report_physics` 的那张表能打出来（SI 单位，不是归一化空间的数）。
- feature map 的 PCA 伪彩色贴回原图对得上边缘（05 号票的判据在这里复查一次）。


---

## 05 号票（2026-09-08 关闭）钉死的实现细节

**照抄，不要重新推导。**

1. **相机：什么都不做。** `trace_cams = list(cam_list)` —— 用场景自己的相机。
   整图 resize 对 `TraceCamera` 是无操作（只存 `FoVx/FoVy`，`focal2fov` 对同比缩放不变），
   而 trace 循环（`trace_instance_to_gaussians.py:1516`）已经把 feature map 插值到相机栅格，
   那一步就是要的映射。**`trace_crop_aligned` 的两个 helper 对 segvggt 都不适用**，
   `create_virtual_crop_camera` 尤其不要碰（它算了 `cx_crop` 却从不传出，偏心裁剪静默错）。

2. **护栏（本票要写的唯一一段几何代码）**：进 forward 前断言
   **全部帧尺寸相同** 且 **`new_h <= new_w`**（横图），否则**硬报错**。
   不满足时 `load_and_preprocess` 会裁剪，恒等映射随之失效 —— 而且不会报错，只会出糊图。
   报错信息要写清是哪张图、什么尺寸。

3. **运行点 = 252×448**，`encoder_batch_size = 4`。两个都不是可调参数：
   252×448 是 01 号票读出"不漂"的那个点，4 是训练视角数且实测单调最优
   （4→0.613、8→0.606、12→0.555、24→0.561）。**不要沿用 iggt 的 48**。

4. **判据里的 GT 用 `sam/mask/*.png`**（原生 756×1008、8 实例），
   **不是** `id_maps/*.npy`（336×504，上一次 IGGT 的产物）。

## Answer

**接上了，bench 上跑通并落盘。`gau_feat [1045236,128]` + `Q_all [93,128]`，
链路的三段（field / query / 物性）实测在同一个空间里。放行 04 号票。**

### 改了什么

| 文件 | 改动 |
|---|---|
| `src/model/arch/segvggt_decode.py` | **新增**。`decode_instances` + `report_physics` 从 `segvggt_infer.py` **原样搬过来**共用（票里要求的"不要重写一遍"）；新抽 `format_physics_table`，两个入口打同一张表。 |
| `scripts/segvggt_infer.py` | 改成 import 上面那个模块；顺带补 `--phys_scheme`（见下）。 |
| `scripts/trace_instance_to_gaussians.py` | 新增 `prepare_segvggt_features` + 三个 helper，注册进 `FEATURE_PREP`；新参数 `--segvggt_ckpt` / `--segvggt_image_size`；`prep` 契约扩了可选的 **`query_bank`**，落盘进同一个 `.pt`。 |
| `scripts/check_segvggt_trace_source.py` | **新增**，本票三条判据的可复现装置。 |

### `query_bank` 的字段（04 号票的输入契约）

`query_proj [M,128]`（**与 field 点积的那个向量**）、`phys_mu_model/phys_var_model [M,3]`、
`phys_si [M,3]`（`physgm_denormalize` 出来的 SI）、`scores [M]`、
`query_idx [M]`（批内 slot id）、`batch_id [M]`、`view_names`、`property_names`。

**只 concat，不合并。** 跨批 slot 身份不通用，所以 `batch_id` 一起存着——
去重是 04 号票的事，这里一个字都不猜。

### 判据逐条

**① bank 与 field 同空间 —— `maxabs = 0.000e+00`，逐位相等。**
拿 `instance_queries_proj(query_embed)` 与 `feature_map` 重做模型自己那个
`einsum("qd,vhwd->qvhw")`，重建出的 `[400,4,126,224]` 与 `pred.query_masks` **一模一样**。
⇒ 我取出来的 `query_proj` **就是**产生 mask logit 的那个向量，不是一个"差不多"的东西。
这是本票最该验的一条：错了不报错，只会让 04 号票在 3D 里点积出垃圾。

**② PCA 伪彩色对得上边缘。** feature 栅格 126×224 vs 原图 756×1008（4.5×/6.0×，
各向异性，正是 05 号票说的整图 resize）。overlay 里墙/地板的分界线沿真实接缝走，
娃娃、猫、蛋挞、玩具车、葡萄各自成块。
顺带复现 05 号票的量化读数：逐视角 mask vs `sam/mask` 的 **mean best IoU = 0.6285**
（05 在同一运行点读到 0.6133，同一量级；我这次用 4 视角单批，05 是另一批）。

**③ 物性表打出来了**，SI 单位：density 610–2112 kg/m³、E 4e7–5e10 Pa、ν 0.29–0.43。
93 行全在 `[segvggt] per-query physics` 那张表里。

### 端到端读数（bench，36 视角）

```
[segvggt] 36 frames, all 1008x756 (landscape) -> 448x252
Tracing 36 views (feat_dim=128, TRACE_CHANNELS=20 x 7 pass(es), backend=surfel)
Gaussians traced: 1,005,708 (96.2%)      Q_all = 93 rows over 9 batches (10.3/batch)
```

9 批与 05 号票预测一致；每批 10.3 个存活 query（05 估的是 ~14，同量级，偏少）。

### 护栏（票里点名要写的唯一一段几何代码）

`_assert_segvggt_preprocess_identity`：**全帧同尺寸** + **横图**，否则硬报错，
报错信息带图名和尺寸。另外 `--trace_crop_aligned` 对 segvggt 直接抛错——
两个虚拟相机 helper 都表达不了这套预处理（05 号票查出 `create_virtual_crop_camera`
算了 `cx_crop` 却不传出）。

`load_segvggt_images` **故意不用** `segvggt_infer.load_and_preprocess`：
后者把高 round 到 14 的倍数、还可能裁。这里是**精确 448×252 的整图 resize**，
恒等假设因此是构造出来的，不是撞上的。

### 顺手补的一个洞（本票范围外一行）

`scripts/segvggt_infer.py` **原来根本加载不了 phys 的 ckpt** ——
它没有 `--phys_scheme`，`_assert_query_physgm_coverage` 必然抛
"Checkpoint contains query_physgm weights but the model was built without
phys_scheme='query_physgm'"，所以它自己的 `report_physics` 一直是死代码。
（`git stash` 验过是原有问题，不是本次重构引入。）补了 `--phys_scheme` 一个参数，
现在两个入口都实际走共用模块。冒烟跑过，物性表和 `physics.npz` 都出来了。

### ⚠️ 带给 04 号票的读数：**2D 的阈值不能直接搬到 3D**

`gau_feat @ Q_all.T` 出得来 `[1045236, 93]`，但**阈值是个真问题**：

| 阈值（归一化场） | 每 query 认领高斯 p50 / max | 覆盖 |
|---|---|---|
| `>0.0` | 6,338 / **859,829**（82% 全场！） | 96.7% |
| `>0.4` | 103 / 96,277 | 10.0% |
| `>0.8` | 0 / 14,792 | 1.5% |

logit 分布 p01=-3.16 / p50=-0.24 / p99=0.31 / max=2.80。
**`>0` 不是可用阈值**（2D 那边 `sigmoid>0.4` ⇔ logit`>-0.405`，搬过来会让某个 query 吞掉全场），
可用区间挤在 0 与 0.4 之间，04 号票必须自己定。

**另一条**：只有**归一化**的 `feat` 能用全局阈值。`feat_unnorm` 每个高斯带一个
`num_ray` 的正倍数（min=1 到 max=1.86e6），符号一样但量级差六个数量级，
所以 `>0.4` 在它上面几乎等于 `>0`（覆盖 89.5%）——**地图等价性论证里那个"和"只保证符号，不保证可比**。

### 复现

```bash
PYTHONNOUSERSITE=1 python scripts/trace_instance_to_gaussians.py \
  -s <scene> -p <scene>/point_cloud.ply --feature_source segvggt \
  --segvggt_ckpt <arm_b_lora.ckpt> --no_trace_crop_aligned -o <out>
SEGVGGT_TRACE_OUT=<out> PYTHONNOUSERSITE=1 python scripts/check_segvggt_trace_source.py
```

`tests/` 71 项全过。

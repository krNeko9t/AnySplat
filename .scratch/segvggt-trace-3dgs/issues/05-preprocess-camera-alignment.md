# 05 — SegVGGT 的预处理与 trace 相机对不对得上

Type: task
Status: closed (2026-09-08) — 整图 resize 是相机的无操作，直接用场景相机
Blocked by: —
Blocks: [03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)
Assignee: claude (wayfinder session, 2026-09-08)

> **这票单独开，是因为它错了不报错。** 预处理与相机差一个 crop，trace 出来的场
> 只是"糊一点"，不会抛异常，然后下游全部结论静默作废。

## Question

`scripts/segvggt_infer.py:57` 的 `load_and_preprocess` 做了三件改变几何的事：

1. resize 到 **宽 518**，高按比例缩放后**round 到 14 的倍数**；
2. 竖图**中心裁成正方形**；
3. 一个文件夹内各帧高度不一致时，**统一裁到最小高度**（`min_h // 14 * 14`）。

而 trace 要用的是场景**自己的相机**（`src/trace_cameras.py` 的 `TraceCamera`，
colmap 或 nerf_transforms 出来的内外参）。feature map 的像素栅格与这套相机的栅格
**不是同一个**：裁过、缩过、还 round 过。

`configs/trace/*.json` 里存在 `"trace_crop_aligned": false` 这个开关，
说明 IGGT 时代就吃过这个坑 —— 要搞清楚它当年到底解决了什么、够不够本图用。

## 要回答的

1. `trace_crop_aligned` 现在的语义是什么？覆盖上面三条里的哪几条？
2. 训练分布是 **252×448**（[phys-on-query 05 号票](../../phys-on-query/issues/05-training-operating-point.md)
   钉死的 `fixed_views_and_shape`），而 `segvggt_infer.py` 默认宽 518。
   **推理该按哪个跑？** 按训练分布跑，分布内但分辨率低；按 518 跑，分辨率高但离训练分布远。
3. 定下一条明确的映射：`(feature map 像素) → (原图像素) → (TraceCamera 内参)`，
   并在 03 号票的实现里落成代码而不是注释。

## 判据

拿 `3dovs/bench` 的一个视角，把 feature map 做 PCA 到 3 通道当成伪彩色，
按定下的映射贴回原图上叠加显示 —— **边界应当对齐到物体边缘**。
歪了一眼就能看出来，这比任何断言都可靠。

---

## Resolution（2026-09-08）

**对得上，而且不用改相机。** SegVGGT 的预处理在本图的两个场景上都退化成"整图 resize"，
而整图 resize 对 `TraceCamera` 是**无操作**——所以 trace 直接用场景自己的相机。

### 为什么是无操作（这条是本票的地基，不是省事）

`TraceCamera`（`src/trace_cameras/trace_camera.py:60`）只存 `FoVx/FoVy`，是个**对称视锥**，
没有主点字段。而 `focal2fov(f·s, W·s) == focal2fov(f, W)` —— 焦距和图宽同比缩放，视场角不变。
⇒ **整图 resize 前后是同一个相机**，只是渲染栅格换了个分辨率。

trace 循环（`trace_instance_to_gaussians.py:1516`）本来就把 `[C,h,w]` 的 feature map
`F.interpolate` 到 `trace_cam.image_height/width`。整图 resize 在**归一化坐标里是恒等**，
所以这一步插值就是票里要的那条映射，**不需要再写任何东西**：

```
feature 格子 (i/h, j/w)  ==  原图像素 (y/H, x/W)  ==  相机视锥里的同一条射线
```

⇒ 票里第 3 问要求的"落成代码而不是注释"，答案是**没有代码可落**。要落的是护栏（见下）。

### 三条把它钉死的事实

1. **bench 不触发任何裁剪。** 36 张图**尺寸全同**（1008×756），且是横图 ⇒
   `load_and_preprocess` 的方形裁剪（`new_h > new_w`）与最小高度裁剪（`im[:min_h]`）两条分支都不进。
2. **主点正好居中。** COLMAP `SIMPLE_RADIAL`，`f=779.258, cx=504.0, cy=378.0` = 恰好 `W/2, H/2`。
   对称视锥对 bench 是**精确**的，不是近似。
3. **feature map 是 `H/2 × W/2`，不是 `H/14 × W/14`。** 252×448 出来是 126×224
   （DPT 头内部已上采样）。分辨率比按 patch 推的高 7×，这条改变了对"糊不糊"的直觉。

### 判据怎么落的（判据在票里已定：PCA 伪彩色贴回原图，看边界）

`prototypes/out/align.png`（三行 = 三个运行点，左伪彩色 / 右 50% 叠加）。
颜色块**分别落在娃娃、猫、毛毛虫、铁盒、玩具车上**，墙/地板的分界线沿着真实接缝走。
**三个运行点全部通过** —— 对齐是映射的性质，与分辨率无关，这正是预期。

### 三个决定

**① 运行点 = 252×448**（训练运行点，各向异性压缩 4:3 → 16:9）。

| 运行点 | feature 栅格 | mean best IoU vs `sam/mask` |
|---|---|---|
| **A 252×448（训练点）** | 126×224 | **0.6133** |
| B 336×448（保长宽比） | 168×224 | 0.6176 |
| C 392×518（`segvggt_infer` 默认） | 196×259 | 0.6112 |

**表选不出来**（差距在噪声内，单视角单批）。**决定性理由是 01 号票**：它的"不漂、放行 trace-first"
是**在 252×448 上读出来的**。换运行点 = 地基票测的不是实际要跑的配置 ⇒ 要么重跑，
要么带一个没验过的前提往下走。17 天，不值得。附带好处：最省，且 V 增大时退化最缓。

**变形不影响几何**：整图 resize 在归一化坐标里恒等，各向异性只让网络看到一张歪的图，
不会让 trace 打错位置。overlay 第一行确实略糊于二三行，但没糊到 IoU 上。

**② 裁剪 = 设护栏报错，不修主点。**

恒等只在"不裁剪"时成立。两条会破坏它的分支：竖图中心方形裁剪（对称，只错 FoV）；
帧尺寸不齐时 `im[:min_h]` —— **只从底边裁**，主点因此偏移 `(new_h - min_h)/2` 像素（**不对称**）。

⇒ segvggt feature source 里加断言：**全部帧同尺寸 且 `new_h ≤ new_w`**，否则硬报错。
不给 `TraceCamera` 加主点：那要动光栅化器的调用面（`get_projection_matrix` 整个是对称视锥），
本图两个场景都用不上，17 天里风险大于收益。**护栏把静默出错变成响声**，正是本票开票的理由。

⚠️ **顺带查出一个坑**：`create_virtual_crop_camera`（`trace_instance_to_gaussians.py:109`）
**算出了 `cx_crop`/`cy_crop`，然后一个都没传出去** —— `TraceCamera` 根本没有主点字段。
现有的裁剪相机对**偏心裁剪本来就是错的**，而且不报错。不影响本图（bench 主点居中、
segvggt 两个 helper 都不用），**只记录，不修**（归 Out of scope 的"不改 rasterizer"）。

**③ `encoder_batch_size` = 4**（03 号票把这个交给本票定）。

| 一次 forward 几个视角 | A 252×448 | B 336×448 | 峰值显存 |
|---|---|---|---|
| **4** | **0.6133** | **0.6176** | 9.3 / 9.6 GB |
| 8 | 0.6057 | 0.5591 | 10.2 / 10.8 GB |
| 12 | 0.5551 | 0.4602 | 10.7 / 11.4 GB |
| 16 | 0.5405 | 0.4516 | 11.1 / 12.1 GB |
| 24 | 0.5611 | 0.4576 | 12.3 / 13.6 GB |

**单调退化**，且显存（≤14 GB）与速度（0.04–0.07 s/view）**都不是约束** ⇒ 喂 4 个不花钱。
IGGT 配置里那个 `encoder_batch_size: 48` 在这条链路上是**有害的**，不要沿用。

### `trace_crop_aligned` 是什么（票里第 1 问）

它是"把相机换成与编码器预处理对齐的**虚拟相机**"的开关，两个 helper：
`create_virtual_crop_camera`（anysplat：短边 448 + 中心裁 448×448）、
`create_virtual_resize_camera`（iggt：纯 resize 到 `iggt_image_size`）。

覆盖票里三条中的**缩放与中心方形裁剪，且只在 FoV 意义上**；主点偏移它表达不了（见上面的坑）。

**对 segvggt 两个 helper 都不适用** —— 直接 `trace_cams = list(cam_list)`。
这与六份 `configs/trace/*.json` 里已经写着的 `"trace_crop_aligned": false` 恰好一致。

### 两件顺路查出、改动别的票的事

1. **`id_maps/*.npy` 不是 GT。** 它是 **336×504**，是上一次 IGGT 跑出来的产物
   （`iggt_idmap.py:559` 写的就是这个目录），所以长宽比（1.5）和原图（1.333）对不上。
   真正的逐视角 GT 是 **`sam/mask/*.png`，原生 756×1008，8 个实例**。
   ⇒ 地图 Notes 里"自带 id_maps GT"已改；[07 号票](07-3dovs-segmentation-metric.md)的指标口径随之改。
2. **`mipnerf360/garden` 的原图找到了** —— 185 张在
   `/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/garden/images`
   （5187×3361，与 `cameras.json` 对得上；`data/official/` 那份只有 ply + json）。
   但那边**没有 COLMAP `sparse/`**，而 `CAMERA_BACKENDS`（`src/trace_cameras/registry.py:41`）
   只认 `colmap` / `nerf_transforms`。⇒ [08 号票](08-3dgs-backend-on-garden.md)要多写一个
   读 `cameras.json` 的加载器，**不是**"colmap 现成"。

### 带给下游的一条量级

一次喂 4 个 ⇒ bench 36 视角 = **9 批**，garden 185 视角 = **47 批**。
每批约 14 个存活 query ⇒ garden 上 `Q_all` 约 **650 行**。
[04 号票](04-query-pooling-and-3d-dedup.md)的 3D IoU 去重要按"几百"设计，不是"几十"。

### 交付

装置捕获在 `research/05-preprocess-camera-alignment` 分支
（`prototypes/check_preprocess_alignment.py` + `prototypes/sweep_views.py` + `out/`），主干不留。

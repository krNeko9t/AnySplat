# 05 — SegVGGT 的预处理与 trace 相机对不对得上

Type: task
Status: open
Blocked by: —
Blocks: [03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)
Assignee: —

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

# 08 — 在真正的 3DGS 场景上验一遍后端

Type: task
Status: open
Blocked by: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Blocks: —
Assignee: —

## Question

本图的主场景 `3dovs/bench` 是 **2DGS**（实测 ply 只有 `scale_0` / `scale_1`，
走 `diff_surfel_rasterization`）。而你原始的诉求是"输入一个 3DGS 场景"。

`resolve_trace_backend`（`trace_rasterize.py:23`）按 ply 的 scale 维度数自动分派，
2DGS/3DGS 在上层只是换个 backend —— **理论上不影响任何结论**。但"理论上"要验一次。

本机实有的 3DGS 场景：
`/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/{bicycle,garden,room}/point_cloud.ply`
（3 个 scale ⇒ 真 3DGS）。**无任何 GT**，所以这票只出定性结果。

## 要小心的

- 相机从哪来？mipnerf360 是 colmap 场景，`camera_backend: colmap` 现成，
  但要确认那份 `sparse/` 与这份 ply 是同一次重建的产物（不是的话坐标系对不上，
  出来的图会整个错位而不报错）。
- mipnerf360 是**户外大场景**，而 ckpt 训在 Infinigen 室内。分布外程度比 3dovs 大得多，
  分割烂掉是**预期内**的 —— 这票的判据是"3DGS 后端跑得通"，不是"分割好看"。
  别把后端验证和泛化能力两件事混着判。

## 判据

`garden` 上跑完不崩、出一张上色图，能看出 backend 走的是 `3dgs` 分支。
分割质量如何**如实记录**，不作为通过条件。

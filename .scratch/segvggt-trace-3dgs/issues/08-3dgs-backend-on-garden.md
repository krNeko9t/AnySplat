# 08 — 在真正的 3DGS 场景上验一遍后端

Type: task
Status: open
Blocked by: ~~[04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)~~（已关闭 2026-09-08，放行）
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

- 相机从哪来？⚠️ **`camera_backend: colmap` 并不现成** —— 根本没有 `sparse/`，
  只有 `cameras.json`（05 号票 2026-09-08 查证，见下）。
  仍要确认那份 `cameras.json` 与这份 ply 是同一次重建的产物（不是的话坐标系对不上，
  出来的图会整个错位而不报错）。
- mipnerf360 是**户外大场景**，而 ckpt 训在 Infinigen 室内。分布外程度比 3dovs 大得多，
  分割烂掉是**预期内**的 —— 这票的判据是"3DGS 后端跑得通"，不是"分割好看"。
  别把后端验证和泛化能力两件事混着判。

## 判据

`garden` 上跑完不崩、出一张上色图，能看出 backend 走的是 `3dgs` 分支。
分割质量如何**如实记录**，不作为通过条件。


---

## ⚠️ 05 号票（2026-09-08）查证：数据在，但相机加载器要新写

- **原图找到了**：185 张在
  `/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/garden/images`
  （5187×3361，与 `cameras.json` 的 `width/height` 对得上）。
  `data/official/mipnerf360/garden/` 那份**只有 `point_cloud.ply` + `cameras.json`**，没有图。
  bicycle / room 同样。
- **没有 COLMAP `sparse/`**，三个场景都没有。而 `CAMERA_BACKENDS`
  （`src/trace_cameras/registry.py:41`）只认 `colmap` / `nerf_transforms`。
  ⇒ 本票要**多写一个 `cameras.json` 加载器**（字段：`img_name / width / height / fx / fy /
  position / rotation`，**没有 `cx/cy`** —— 与 `TraceCamera` 的对称视锥恰好相容）。
  这是本票的第一件事，不是顺带的事。
- **预处理护栏在这里会说话**：garden 是横图（5187×3361，1.543）且尺寸全同 ⇒
  03 号票那条护栏通过，仍然是"整图 resize、相机无操作"。
  按 448 宽算高：`448/1.543 = 290.3 → round14 → 294`。
- 01 号票的第三条待办（"08 顺手在室外/3DGS 上复跑一次漂移脚本"）仍然有效，
  脚本换 `--image_dir` 即可。


---

## 04 号票（2026-09-08 关闭）留给本票的三条

1. **`query_bank` 多了 `mask_frac` 字段**（每个候选 2D mask 的面积占比），
   重跑 trace 会自动带上。旧的 `.pt` 没有它，`segvggt_pool_instances.py` 会告警并跳过
   2D 过滤——那时背景 slot 会吞掉整个场景，**别当成 3DGS 后端的 bug**。
2. **背景 slot 是不是也是 234，本票是第二个样本。** bench 上 9 个批全是 slot 234，
   2D 面积 0.492–0.769，真物体最大 0.126。garden 上要**复看这个空档还在不在**——
   在，`--max_mask_frac 0.3` 就是个稳的默认值；不在，就得靠 `--max_scene_frac` 那条 3D 兜底。
3. **「成员数 / 覆盖批数」这个免费置信度，garden 才是它的检验场。**
   bench 9 个批上真物体是 10–15 成员跨 9 批、假阳性是单成员单批。garden **47 批**，
   如果这个规律还成立，它就比 score 更好用；成立与否请带数字回来（04 没敢拿它当阈值）。

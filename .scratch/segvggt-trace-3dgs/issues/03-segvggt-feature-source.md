# 03 — 把 SegVGGT 接成 trace 脚本的一个 feature source

Type: task
Status: open
Blocked by: [02 — 让 trace 支持 feat_dim > 20](02-chunked-trace-over-20-channels.md), [05 — 预处理与 trace 相机对不对得上](05-preprocess-camera-alignment.md)
Blocks: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Assignee: —

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

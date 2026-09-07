# 11 — depth/pose 漂移的只读指标

Type: task
Status: open
Blocked by: —

> 由 [01 号票](01-unfreeze-lora.md) 的 Q2 判决毕业而来。

## Question

01 号票查明：`camera_head`/`depth_head` 全冻 + `segvggt_geo.weight: 0` ⇒
**没有任何 loss 在盯几何，也没有任何指标在报它**。所以放开 LoRA 后底座一动，
几何漂没漂**根本看不见**——不是"漂了会被罚"，是"漂了没人知道"。

01 号票已判：**几何不作为本图交付**（终点只写 class + P，字面不含几何），
因此不给几何加 loss。但要给它加**只读指标**。

## 要做的

在 `SegVGGTWrapper` 的 validation 里报出 depth / pose 误差，**只记录、不进 loss**，
`segvggt_geo.weight` 保持 0。计算逻辑已经现成——`src/loss/loss_segvggt_geo.py`
（`LossSegVGGTGeo`：camera 是 9 维 pose encoding 上的 Huber；depth 是按序列单一 median
scale 对齐后的 L1+梯度），wrapper 已经把 `pred_pose_enc_list` / `depth` / `geo_target`
塞进 `depth_dict`。本票是把它在 `torch.no_grad()` 下当指标算一遍并 log，
不是新写一套。

## 为什么值得做

它是 01 号票那张两臂对照表的**第二列**：第一列答"放开 LoRA 分割/物性涨不涨"，
第二列答"代价是不是几何塌了"。没有第二列，(b) 臂就算赢了也不知道赢在哪、赔了什么。
成本是几行代码。

## 完成判据

(a)/(b) 两臂的 val 日志里都能读到 depth / pose 误差曲线；`segvggt_geo.weight` 仍为 0；
该配方的 lock **不变**（只读指标不碰任何 `requires_grad`）。

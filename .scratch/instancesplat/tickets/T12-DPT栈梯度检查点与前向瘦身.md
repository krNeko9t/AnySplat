---
id: T12
title: DPT 栈梯度检查点与前向瘦身
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

R3 的结论：8 视角 @448 全量微调峰值 **47–52 GB**，40G 卡缺口 9–14 GB；
**唯一能一次性解决问题的开关是给四个 DPT 栈加梯度检查点**（把激活 24–27 GB 压到 ~6–9 GB，
总峰值降到 22–26 GB）。不做这个就只能退到 6 视角，而 6 视角本身也还在 40–43 GB，仍越界。

> **T8 实测锚点**：`voxelize_ratio` = **0.906 / 0.840 / 0.769 @N=2/4/8**
> （N=8 逐场景 0.504–0.934），即体素化只压缩 8–28%。
> N=8 时 GS = 0.98M（spp 336×448）/ 0.73M（re10k 280×448），硬上界 1.20M
> ——**显存预算不能指望体素化压缩**。零样本推理（b=1、backbone 冻结、bin 分辨率）
> 峰值 3.77 / 4.49 / 6.08 GiB @N=2/4/8，是「静态 + 激活」的冻结下界锚点。

现状（已在代码里查实）：

- `use_checkpoint = True` **硬编码**在 `src/model/vggt/models/aggregator.py:71` 与
  `src/model/vggt/layers/vision_transformer.py:97`；`src/model/vggt/heads/camera_head.py:137`
  直接调 `torch.utils.checkpoint.checkpoint`。→ 909M transformer 的激活已被压到 ~1.2 GB。
- `grep -n checkpoint src/model/vggt/heads/dpt_head.py` **返回空**。四个 DPT 栈
  （`depth_head` / `point_head` / `gaussian_param_head` / `PartHead`）**一个都没开**，
  而它们占激活的 18/24 GB。

## 要做的

1. **给 DPT 栈加梯度检查点**。切分粒度要选（每个 refinenet block？整个 head？），
   给出选择理由和实测/估算的显存-算力权衡。**新增的开关默认值必须让既有实验行为不变**
   （地图硬约束：不干涉既有实现）。
2. **去掉重复的 point_head**。`anysplat.py:131-142`：`pred_head_type: depth` 时构造
   `depth_head`，随后 `_use_point_head_for_part`（`instance_feat_dim > 0` 时为真）**又**构造
   `point_head`，于是两个 DPT 栈同时跑。R3 估算这里能省 ~2 GB。
   注意 `anysplat.py:496` 的 `point` 分支因 `depth_conf` 未定义是坏的——**先看清楚
   `_use_point_head_for_part` 到底要的是 point_head 的哪几个中间特征（out2/out3/out4）**，
   再决定是复用 depth_head 的中间特征还是保留一个精简的 point_head。别为了省显存改坏 PartHead 的输入。
3. 汇总 R3 findings 里其余的省显存开关，逐项判断本票做不做、还是留给 T7 配置层：
   ZeRO-1、`gradient_as_bucket_view=True`、DPT/LPIPS 走 bf16。

   > **T11 追加的硬依赖：ZeRO-1 在本票里不再是「可选项」，是必做项。**
   > T11 已把 `aggregator` 的参数 dtype 改成可配（`aggregator_param_dtype`，默认仍 `bfloat16`）。
   > 但地图的配方要求 backbone 解冻，而 bf16 参数下实测 **68.5%（≈623M）的 backbone 参数
   > 在 `lr=2e-5` 下永远不更新**——所以 T7 的新 experiment yaml **必然**要开 `float32`。
   > 开了就是**静态项 +7.27 GB**。R3 算过：配上 ZeRO-1 后静态项 = 11.25 GB，
   > 反而低于现状的 12.7 GB；**不配 ZeRO-1 就直接开 fp32 会把 R3 的 ✅ 结论推翻**。
   > 详见 [T11](T11-修复aggregator-bf16强制转换.md) 的「五、留给别的票的两条」。
4. **澄清并记录**：`frames_chunk_size` 在训练时**完全不省显存**（每块激活都要留给 backward），
   它只对推理有效。这条要写进 `## 解决`，免得后面有人拿它当省显存手段。

## 硬约束

`instseg_anysplat.yaml`、`loss_disc.py`、`loss_mvc.py`、`config/model/encoder/anysplat.yaml`
的既有字段语义不动。SegVGGT 整条线不碰（注意 `src/model/segvggt/heads/dpt_head.py` 是另一份，
**不要顺手一起改**）。

## 验证（CPU 合成张量）

- 开关关闭时，前向输出与开启前**逐比特相同**（既有行为不变）。
- 开关打开时，前向输出数值一致（在 checkpointing 的重算误差容限内），且梯度能正常回传到
  `PartHead` 与 aggregator。
- 用 `torch.utils.checkpoint` 时确认 `use_reentrant=False`（与 `camera_head.py:137` 一致），
  否则和 DDP / 多次 backward 会打架。
- 去掉重复 point_head 后，PartHead 的输入张量 shape 与数值与改动前一致。

**显存的真实数字必须在集群上测**——本机没 GPU。本票交付代码 + CPU 正确性测试；
峰值显存的实测归 T9 smoke run（或 T8 顺手带一次探针）。

# 11 — depth/pose 漂移的只读指标

Type: task
Status: closed (2026-09-07)
Blocked by: —
Assignee: @krNeko9t (session 2026-09-07)

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

## Answer（2026-09-07 实做，GPU 服务器 `bms-39468022-001` / env `anysplat`）

**做完了，两处改动，六个指标，lock 未动。**

### 改了什么

1. `src/loss/loss_segvggt_geo.py` — 新增 `LossSegVGGTGeo.metrics(depth_dict)`，
   `@torch.no_grad()`，返回 detach 过的 dict。**不是新写一套**：直接调用
   `_camera_loss` / `_depth_loss`，所以 `geo_camera` / `geo_depth` 与
   `loss_segvggt_camera` / `loss_segvggt_depth` 同量纲、可并排看。
2. `src/model/wrapper/segvggt_wrapper.py` — `_log_geo_drift()`，在 `validation_step`
   里 `depth_mean` 之后调用一次；`__init__` 里取 `self.losses` 里现成的
   `LossSegVGGTGeo` 实例（取不到就 default 构一个，指标不能依赖 geo loss 在不在 loss 列表里）。

### 报出来的六个数

| 指标 | 含义 |
|---|---|
| `val/geo_camera` | 全部 refinement 迭代 + gamma 衰减，= loss 那一项 |
| `val/geo_camera_T_last` / `_R_last` / `_fl_last` | **末次**迭代的平移 / 旋转 / 焦距分解 |
| `val/geo_depth` | scale 对齐后的 L1 + 梯度项，= loss 那一项 |
| `val/geo_depth_rel` | 同一个数 ÷ GT depth 中位数 |

分解项是票面之外加的：没有 loss 盯着的时候，"漂了"只有同时说出**漂的是什么**才有用。

### 为什么加了 `geo_depth_rel`（实测逼出来的）

首跑读到 `val/geo_depth = 106.3`，看着像塌了。查了一遍 **GT depth 是毫米**
（scannetpp PNG 原样进来，未除 1000；实测 val 样本 median **2263**、p1 1583、p99 4071）
⇒ 106 mm 是 **4.7% 相对误差**，冻结的 depth 头其实好得很。
绝对值带着数据集单位，跨臂/跨数据集不可比，所以并排报一个无量纲的比值。

**顺带一条给以后的人**：训练侧 `loss_segvggt_depth=117.6` vs `loss_segvggt_camera=0.0223`,
配方是 `lambda_camera: 5 / lambda_depth: 1` ⇒ 真要把 `segvggt_geo.weight` 打开，
depth 会以约 **1000×** 压死 camera 项。现在 `weight: 0` 挡着，无害；本图不开 geo loss，
**不在本票内处理**，记在这里免得下次被咬。

### 验证（都实跑过）

- 合成张量单测：六个 key 齐全、全部 `requires_grad=False`；`geo_camera` / `geo_depth`
  与 `weight=0` 下 `forward()` 的 `extra_logs` **逐位相等**（差 <1e-5）；
  `forward()` 仍返回 **0.0**（loss 未被污染）；constant depth 目标（manifest 的无深度占位：
  depth 全 1 / valid 全 True）自动**丢掉** depth 两项而不是报 0；
  `metrics(None)` / `metrics({})` / 全 None target 均返回 `{}` 不炸。
- 真跑一遍 `+experiment=segvggt_agnostic_phys_joint trainer.max_steps=1
  trainer.val_check_interval=1`（1×A100，约 1 分钟）：
  `[val geo step=1] camera=0.0079 (T=0.0134 R=0.0001 fl=0.0022) depth=106.308 depth_rel=0.0521`，
  tfevents 里 `val/geo_*` 六条全在。
- `Trainable params: 487 M` 不变；`scripts/freeze_lock.py +experiment=segvggt_agnostic_phys_joint`
  ⇒ **`segvggt_agnostic_phys_joint.lock: unchanged`**；`segvggt_geo.weight` 仍为 0。

### 完成判据对账

`weight: 0` ✅ · lock 不变 ✅ · val 日志读得到 depth/pose 曲线 ✅。
"两臂都能读到"——指标在 **wrapper 里**，不在 config 里，**两臂自动都有**；(b) 臂的 config
本身还没建（等 [09](09-freeze-matching-language.md) / [05](05-training-operating-point.md)），
这一格由 05 收口，不由本票。

### 现场捡到的环境地雷（不属本票，已记进 map Notes）

`anysplat` env 当场 `import` 就炸：`pytorch3d._C` / `torch_scatter` undefined symbol。
根因不是 env 坏了——是 **`~/.local/lib/python3.10/site-packages` 里有一个 2026-09-05 装的
torch 2.7.1+cu118**，user-site 优先级高于 env，把 env 自己的 **torch 2.4.1+cu124** 顶掉了，
而两个编译扩展是按 pt24cu124 编的。**`PYTHONNOUSERSITE=1` 一加即好**，
本票所有实跑都带着它。不加的话训练根本起不来——直接挡着 [05 号票](05-training-operating-point.md)。

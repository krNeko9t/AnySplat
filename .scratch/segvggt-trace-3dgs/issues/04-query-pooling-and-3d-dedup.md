# 04 — 跨批 query 池化 + 3D IoU 去重

Type: task
Status: resolved
Blocked by: ~~[01 — SegVGGT 的 feature map 跨 forward 漂不漂](01-cross-batch-feature-consistency.md)~~, ~~[03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)~~（均已关闭 2026-09-08）
Blocks: [06 — 物性数值怎么摆才不撒谎](06-how-to-present-physics-honestly.md), [07 — 3dovs 上顺路报分割指标](07-3dovs-segmentation-metric.md), [08 — 在真正的 3DGS 场景上验后端](08-3dgs-backend-on-garden.md), [09 — 验收交付什么](09-acceptance-artifacts.md)
Assignee: krNeko9t

> 这是把"一堆碎片"变成"一堆实例"的那一步，也是本图 Destination 的主体。

## Question

03 号票跑完，手上是：全局的 `gau_feat [P,128]`，和 M = Σ_b N_b 个候选 query
（每行自带 128 维投影、score、和它那一行物性）。M 在 25 批 × ~15 的量级 ≈ 几百。

单批 query 不覆盖场景（一批 4 视角只含那 4 个视角里的物体，地图核心论证），
所以要把所有批并起来再去重：

```
Q_all  = concat_b(Q_b)                    # [M, 128]
logits = gau_feat @ Q_all.T               # [P, M]
```

`logits` 每一列是一个候选实例在**全部** P 个高斯上的响应——包括它自己那批视角
从没见过的高斯。然后在 3D 里合并。

## 要定的事（本票要答的，不是已知的）

1. **高斯归属怎么定**：逐列 `sigmoid > thr` 得到软集合，还是逐行 `argmax` 得到硬划分？
   前者允许重叠和"无人认领"，后者保证划分完备但强行给每个高斯派一个主。
   建议先做前者，"无人认领"的比例本身就是要报的数（→ 地图 Not yet specified）。
2. **合并判据**：候选之间按**高斯集合 IoU** 合并（阈值待定）。
   备选是按 `Q_all` 的 embedding 余弦合并 —— 但那等于又信了一遍跨批一致性，
   而 IoU 是在 3D 里直接看重叠，独立于 01 号票的结论。**倾向 IoU。**
3. **合并组的物性怎么算**：按 score 加权平均 mu？还是取 score 最高那一行？
   `var` 怎么合？（注意 `var` 是学出来的预测方差，不是 GT 方差的回归 ——
   见 `query_physgm_readout.py` 的 docstring，别当成置信区间乱用。）
4. **归一化**：`gau_feat = sum_gau_sem / num_gsem.clamp(min=1)`。
   `num_gsem` 很小的高斯（只被一两条光线打到）要不要直接丢？

## 判据

`3dovs/bench` 上出一张 3D 分割上色图 + 一张实例列表（实例 id / 高斯数 / score / E,ν,ρ）。
**先看图像不像话，再谈指标**（指标归 07 号票）。


---

## 05 号票（2026-09-08）带来的量级

`encoder_batch_size` 定死 **4** ⇒ bench 36 视角 = **9 批**，garden 185 视角 = **47 批**。
每批约 14 个存活 query ⇒ garden 上 `Q_all` 约 **650 行**。
3D 高斯集合 IoU 去重要按 **650×650** 设计，不是"几十×几十"。


---

## Answer

**跑通了，bench 上出图。93 个跨批候选 → 过滤掉 1 个背景 slot（在 9 个批里都是它）
→ 77 个 → 按高斯集合 IoU 合并成 8 个实例，每个由 10–15 个候选、跨全部 9 个批组成。
定性图上娃娃 / 泰迪熊 / 葡萄 / 蛋挞 / 玩具车 / 碗各自成块，换视角颜色不变。**

`3dovs/bench` 的 GT 是 8 个实例，这里也是 8 个（没做匹配，指标归 07 号票）。

### 改了什么

| 文件 | 改动 |
|---|---|
| `scripts/segvggt_pool_instances.py` | **新增**，本票的主体：池化 → 过滤 → 3D IoU 合并 → 物性池化 → 实例表 + 逐高斯标签 + 上色 PLY + 对比渲染。 |
| `scripts/check_query_pooling.py` | **新增**，四个选择各自的对照数据（下面每条读数的可复现装置）。 |
| `scripts/trace_instance_to_gaussians.py` | `query_bank` 加一个字段 `mask_frac`（每个候选 2D mask 的面积占比）。理由见下面的「本票最重要的发现」。 |

### 本票最重要的发现：吞掉全场的不是阈值问题，是**一个可点名的背景 slot**

03 号票留了个警告：`logit > 0` 时某个 query 认领 82% 的全场，所以「2D 的阈值不能搬到 3D」。
查下来那不是阈值的问题——**是 slot 234**，而且是**全部 9 个批里的 slot 234**。
它在 2D 就已经是背景了：batch 0 里它的 mask 盖住 **67.5% 的画面**，而同批其它 10 个候选
最大只有 9.8%。它是 DETR 那个专门吃 "stuff" 的槽位。

| | 2D mask 面积 |
|---|---|
| slot 234（9 个批全有） | **0.492 – 0.769** |
| 其余全部 84 个候选 | ≤ **0.126** |

两者之间是 4× 宽的空档 ⇒ `--max_mask_frac 0.3` 不是调出来的数，是掉在空档里的。
**这件事在 2D 就能判，不需要为它去调 3D 的阈值**，所以 `mask_frac` 加进了 03 号票的
`query_bank` 契约（decode 时顺手算，一行）。另配一个 3D 兜底
`--max_scene_frac 0.2`（认领超过 20% 传出高斯的候选也算 stuff）——bench 的 batch 2
那个 234 读到 0.492，离 2D 阈值只差 0.008，靠 2D 一条腿站不稳，08 号票换场景更靠不住。

⚠️ 注意别过度解读：**slot 234 跨批稳定，只说明有一个槽位全局特化成了背景**，
不等于 slot 身份对物体也跨批通用。地图核心论证（DETR slot 身份不跨批）不受影响。

### 四个选择，逐条给对照

**① 高斯归属：soft 集合（`logit > 0`），不是逐行 argmax。** 保住「无人认领」这个数
（票里要求的）。代价比预想的小：已认领的 128,033 个高斯里只有 **7.4% 被 ≥2 个实例同时认领，
最多叠 3 层** ⇒ soft 集合本来就几乎是个划分，argmax 只在上色时当 tie-break。

**② 合并判据：高斯集合 IoU，不是 query embedding 余弦。** 票里「倾向 IoU」的理由是
独立于 01 号票；实测还多两条：

- **余弦复现不了 IoU 的分组**：最好也只有 **F1 = 0.842 @ cos > 0.80**（漏合 150 对、错合 96 对）。
- **余弦不稳**：阈值 0.5→0.9 实例数 **6 → 7 → 8 → 10 → 17**；IoU 阈值 0.15→0.5 只在 **6 → 9** 之间走，
  选定的 0.3 在平台中间。

这与 01 号票带下来的第①条（query 比 feature 漂得多，别用 query 余弦去重）**独立地对上了**。

**③ 阈值 = `logit > 0`，就是模型自己的 2D 边界，不是调出来的。**
2D 上判定边界本来就压在 0：正样本 `cos(q,f)` p05 = **0.000**，负样本 p99 = **-0.017**。
trace 是正权重加权平均 ⇒ 只压模长不动方向（`|f|` 2D p50 = 3.46 → 3D p50 = 0.0425，**~80×**），
**能扛过这次缩放的只有符号**。所以「2D 阈值搬不到 3D」这件事，去掉背景 slot 之后就不成立了：
每候选认领的最大高斯数从 **85.5% 掉到 8.1%**。

**④ `num_ray` 小的高斯：不丢。** 票里担心它们是噪声。实测反过来——命中越少认领**越少**：

| num_ray | 高斯数 | 被认领 | 被多个实例认领 |
|---|---|---|---|
| [1,5) | 10,267 | 6.0% | 0.14% |
| [5,20) | 34,965 | 5.5% | 0.07% |
| [20,100) | 136,173 | 5.4% | 0.05% |
| [100,1000) | 548,497 | 11.1% | 0.69% |
| [1000,∞) | 275,806 | 20.8% | 2.01% |

丢掉它们只减 recall，不减噪声 ⇒ `--min_num_ray 1`（= 被 trace 到过就算）。

### 物性怎么合（票里的第 3 问）

**在 model space 里按 score 加权平均，再 denormalize。** 不是在 SI 里平均：
`physgm_denormalize` 对密度和 E 是 `10 ** (x·std + mean)`，model space 就是对数域，
在那里平均等于 SI 里的加权几何平均——这才是「同一个物体的若干次估计」该待的地方。

`var` 按同样的权重平均，但**单独摆**，不和 `mu_spread`（成员之间 mu 的标准差）混：
前者是读出头预测的方差（`query_physgm_readout.py` 的 docstring 说了它不是任何 GT 方差的回归），
后者才是「几个批彼此同不同意」的诚实读数。两个都落进 `instances.json`
（`*_var_model` 和 `*_hi_1sigma`），怎么摆归 06 号票。

### 端到端读数（bench，36 视角，9 批）

```
1,045,236 高斯 x 93 候选 → 2D 面积过滤 -9 → 尺寸过滤 -7 → 77 候选 → IoU>0.3 → 8 实例
已认领 128,033 (12.2%)   无人认领 917,203 (87.8%)
```

| id | #高斯 | score | 成员 | 批 | density | E | ν |
|---|---|---|---|---|---|---|---|
| 5 | 80,511 | 0.91 | 10 | 9 | 734.8 | 2.59e9 | 0.348 |
| 3 | 22,077 | 0.95 | 10 | 9 | 1628 | 2.38e10 | 0.315 |
| 2 | 9,107 | 0.95 | 15 | 9 | 861.8 | 2.77e9 | 0.334 |
| 1 | 5,834 | 0.97 | 10 | 9 | 745.8 | 6.66e9 | 0.316 |
| 0 | 4,389 | 0.98 | 10 | 9 | 1094 | 7.14e9 | 0.311 |
| 4 | 3,583 | 0.94 | 13 | 9 | 720.8 | 5.49e7 | 0.416 |
| 7 | 2,151 | 0.38 | 1 | 1 | 698.8 | 5.22e7 | 0.412 |
| 6 | 381 | 0.89 | 6 | 6 | 1095 | 5.90e9 | 0.352 |

**「成员 10–15 个、横跨全部 9 个批」是本票最强的旁证**：同一个物体在每一批里都被提出来一次，
而且这些提议在 3D 里落到了同一堆高斯上（IoU > 0.3 才会被合）。这是 01 号票的结论
在 3D 里的独立复现——01 量的是特征余弦，这里量的是高斯集合重叠。

图在 [`assets/04/`](../assets/04/)（`02/24/26_compare.png`，左原图 / 右实例上色，
灰 = 无人认领；同目录还有 `instances.txt` / `instances.json`）。换视角同一物体同色。
trace 的 `.pt`（1.6 GB）不进 git，`out/` 已加进 `.scratch/segvggt-trace-3dgs/.gitignore`。

### 带给下游的三条

① **「无人认领 87.8%」基本就是背景**（墙 / 地板 / 桌面）——被过滤掉的 slot 234 认领的正是这一块。
地图 Not yet specified 里「没被认领的高斯怎么处理」因此有答案的雏形了：
**报出来当 recall 缺口，不要塞进第 0 类**（塞进去等于偷偷把背景算成一个实例）。
但 id 5 那个 80,511 高斯的实例在图上是**碎成片的木地板**，说明 stuff/thing 的界线
并没有被 `max_mask_frac` 一刀切干净——07 号票算指标时要留意这一个。

② **id 7（score 0.38、单成员、单批）是个假阳性的形状**：真物体都是 10+ 成员跨 9 批。
「成员数 / 覆盖批数」本身就是一个免费的置信度，比 score 更能筛——但**不要现在就拿它当阈值**，
bench 一个场景不够，等 08 号票在 garden 上（47 批）看过再说。

③ **`mask_frac` 进了 `query_bank` 契约**，08 号票复跑 trace 时会自动带上；
旧的 `.pt` 没有这个字段，脚本会告警并跳过 2D 过滤（这时背景 slot 会吞掉场景，别当成 bug）。

### 复现

```bash
PYTHONNOUSERSITE=1 python scripts/trace_instance_to_gaussians.py \
  -s <scene> -p <scene>/point_cloud.ply --feature_source segvggt \
  --segvggt_ckpt <arm_b_lora.ckpt> --no_trace_crop_aligned -o <out>
PYTHONNOUSERSITE=1 python scripts/segvggt_pool_instances.py \
  --traced <out>/gaussian_segvggt_feat.pt
SEGVGGT_TRACE_PT=<out>/gaussian_segvggt_feat.pt \
  PYTHONNOUSERSITE=1 python scripts/check_query_pooling.py
```

`tests/` 71 项全过。

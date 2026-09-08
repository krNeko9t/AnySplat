# 01 — 物性怎么从 dense feat 走到 3D 实例（推理侧不能吃 GT mask）

Type: grilling
Status: open
Blocked by: —
Blocks: [02 — 物性头训练接线与运行点](02-train-physics-head.md), [04 — trace 两路特征](04-trace-two-feature-sources.md)
Assignee: —

> **本图的核心设计票，也是关键路径的第一环**：02 号票（起跑训练）等着它，
> 而训练是本图唯一实打实的时间。地图 Notes 说的「01 → 02 今天做完、今晚起跑」就是这张票。

## Question

`physgm_dpt` 是实例级物性头，但它**训练时吃 GT instance mask**
（`scheme.py:43` 的 `PhysicsSchemeInputs.instance_mask`）。推理时我们没有 GT mask，
只有 HDBSCAN 在高斯上聚出来的实例。**这条失配怎么处理**，决定了训练要怎么配（02）
和 trace 要搬什么（04）。

### 候选 A（倾向）：3D 池化，头不动

`physgm_dpt` 的头是「dense feat → **掩码平均** → per-property MLP」，
**池化严格在 MLP 之前**（`physgm_dense_readout.py:68-90`：
`pool_one_sample(feat_map[b], instance_mask[b])` → `pooled [K,C]` → `decoder(pooled)`）。

⇒ 推理时：把 32 维 dense feat **trace 到高斯上** → 按 HDBSCAN 实例的高斯集合**在 3D 里平均**
→ 过**同一个 MLP** → `(mu, var)` → `physgm_denormalize`。**头一个字节都不用改。**

**要判的是这条等价链站不站得住**，两处不等价必须量出来、写清楚：
1. **池化口径不同**：训练是「逐像素、跨视角按像素数加权」，推理是「逐高斯」。
   trace 出来的 `feat = sum_gau_sem / num_ray` 已经做了一次归一化，
   两者的加权不是同一个（老图 04 号票记过：`num_ray` 小的高斯认领得少）。
2. **成员集合不同**：GT mask 圈的像素 vs HDBSCAN 聚的高斯。
   这一条不可能消除，只能量它有多大——**可查**：在 bench 上同时用
   「GT mask 反投影到高斯」和「HDBSCAN 实例」两种成员集合过同一个 MLP，比两组 (E, ν, ρ)。

### 候选 B：训练时就用预测 mask

把训练侧的 `instance_mask` 换成 IGGT 自己 forward 出来的实例（训练时也跑一遍聚类）。
消除了失配，但**聚类进训练循环**，代价和不稳定性都上去了，两天内不合适。

### 候选 C：逐点物性场，彻底不要实例池化

改头，让它逐像素输出 (mu, var)，训练时把实例级 GT 广播到该实例的所有像素。
推理时物性直接 trace 到每个高斯上，实例只是事后聚合的一个视图。
**表示上最干净**（连"物性在 3D 空间的粒度"这条老 fog 都一起答了），
但**要改头 + 改 loss**，且 GT 广播会让每个物体内部的监督完全均一 —— 不是两天的活。

## 还要一起定的

- **32 维要不要降**。`phys_feat_dim: 32` ⇒ trace 2 趟（G5）。降到 20 以下能省一趟，
  但会动头的结构、要重训。倾向：**不降**，168.5 ms/view 的分趟成本对 bench 36 帧是 6 秒。
- **物性挂在哪个空间做平均**：model space（z-score 后的对数域）还是 SI。
  老图 04 号票在 SegVGGT 那条路上定的是 **model space 按 score 加权平均再 denormalize**，
  这里没有 score，只有高斯数 —— 是等权还是按 `num_ray` 加权？
- **var 怎么报**。老图 04 定的是「`var` 与成员间 `mu_spread` 分开摆」，这里同样适用。

## 完成判据

一句话结论：推理侧的物性池化走哪条路；以及若走 A，那两处不等价各有多大（带控制组的读数，
⚠️ 地图 F4：trace kernel `atomicAdd`，别用 `torch.equal`）。
**不要求在本票内实现 trace 侧**（那是 04），但要给 02 一个明确的「训练侧照原样配即可 / 要改什么」。

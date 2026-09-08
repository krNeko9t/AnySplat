# 01 — 物性怎么从 dense feat 走到 3D 实例（推理侧不能吃 GT mask）

Type: grilling
Status: closed
Blocked by: —
Blocks: [02 — 物性头训练接线与运行点](02-train-physics-head.md), [04 — trace 两路特征](04-trace-two-feature-sources.md)
Assignee: orchestrator (2026-09-09)

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

### 候选 B：训练时就用预测 mask

把训练侧的 `instance_mask` 换成 IGGT 自己 forward 出来的实例（训练时也跑一遍聚类）。
消除了失配，但**聚类进训练循环**，代价和不稳定性都上去了，两天内不合适。

### 候选 C：逐点物性场，彻底不要实例池化

改头，让它逐像素输出 (mu, var)，训练时把实例级 GT 广播到该实例的所有像素。
表示上最干净，但**要改头 + 改 loss**，不是两天的活。

---

## Answer（2026-09-09 关闭）

### 走 **候选 A**：3D 池化，头一个字节不动。训练侧照原样配即可。

**结论一句话**：推理时把 32 维 dense feat trace 到高斯上，按 HDBSCAN 实例在 3D 里
**按 `num_ray` 加权平均**，过同一个 per-property MLP，最后 `physgm_denormalize` 回 SI。

### 判 A 成立的三条现场证据

1. **池化严格在 MLP 之前，且中间没有任何依赖 2D 网格的算子。**
   `physgm_dense_readout.py:68-73` 先 `pool_one_sample(...)` 得到 `pooled [K,C]`，
   `:88-89` 才 `decoder(pooled)`。`decoder` 是
   `LayerNorm → Linear(32,64) → GELU → Linear(64,2)`（`physgm_readout.py:27-37`），
   **逐行都是对 C 维作用的**，对 K 这一维完全 permutation-equivariant，
   也完全不关心这 K 行是从像素来的还是从高斯来的。
   ⇒ 「换成员集合」这件事在数学上就是换 `pooled` 的一行，头不需要知道。

2. **decoder 的第一层是 `LayerNorm`，它把「2D 池 vs 3D 池」的一阶尺度差吃掉了。**
   这是 A 比预期更稳的原因，也是我把 A 从「凑合」升格为「首选」的依据：
   两种池化只要方向一致，模长上的系统差被 LN 归一化掉，只剩方向差进 MLP。

3. **训练侧的池化口径是「跨视角、逐像素等权」**，不是别的：
   `physics_pool.py:79-80` — `masked.sum(dim=(0,2)) / count`，
   其中 `count = (mask_flat == inst_id).sum()` 是**该实例在所有视角上的像素总数**。
   没有 per-view 归一化，也没有面积加权。这一条决定了下面的加权口径。

### 因此定死的四件事

**(a) 3D 侧用 `num_ray` 加权，不是等权。**
trace 出来的每个高斯的特征是 `feat_g = sum_gau_sem_g / num_ray_g`，即
**该高斯被打中的所有光线上的像素特征的均值**。于是

```
Σ_g num_ray_g · feat_g / Σ_g num_ray_g  ≈  该实例所有像素特征的均值
```

——**正好是训练时的口径**（`physics_pool.py:79-80`）。等权（`mean_g feat_g`）则不是：
它把只被 3 条光线擦到的边缘高斯和被 4000 条光线覆盖的主体面片摆成一样重，
训练时从来没有这么池化过。
⇒ **`num_ray` 加权是唯一有训练侧对应物的口径，落地用它。**
等权版本**一并算出来当控制组**（同一份 `.pt`，两行数，成本为零），
用来量地图 01 里说的「不等价 1」。

**(b) 在 model space（z-score 后的对数域）里池化，最后一步才 denormalize。**
准确说：池化发生在**特征空间**（32 维 dense feat），MLP 只跑一次，
出来的 `(mu, var)` 已经在 model space，`physgm_denormalize` 是最后一个算子。
**不要**逐高斯 decode 再平均 SI 值——那会把 MLP 的非线性和 denormalize 的
`exp` 一起搬到平均的错误一侧，且与训练口径不符。

**(c) 32 维不降。**
`phys_feat_dim: 32` > `TRACE_CHANNELS = 20` ⇒ 物性 2 趟 + instance 8 维 1 趟 = 3 趟。
168.5 ms/view × 36 帧 ≈ **6 秒**（bench），garden 185 视角 ≈ 31 秒。
为省这 6 秒去动头的结构、重训一次，是拿关键路径换零。**不降。**

**(d) `var` 与 `mu_spread` 分开摆，`mu_spread` 在本图里有了精确定义。**
- `var` = 头自己输出的 `softplus(out[...,1]) + 1e-2`，是**模型对该实例的不确定度**；
- `mu_spread` = 把该实例**每个高斯单独**过一遍同一个 MLP 得到的 `mu` 的标准差，
  是**成员一致性**。MLP 是 32→64→2 的小网络，10⁶ 高斯 × 3 属性逐点跑一遍是白送的。
两者含义不同，交付表里两列，**不许合并成一个数**。

### 两处不等价：怎么量、在哪张票量

A 不消除失配，它把失配**挪到可量的位置**。两处，各有归属：

| # | 不等价 | 怎么量 | 归属 |
|---|--------|--------|------|
| 1 | **池化口径**：2D 逐像素等权 vs 3D 逐高斯（`num_ray` 加权 / 等权） | 同一张 mask 下，2D 池化得到的 32 维向量 vs 3D 池化得到的 32 维向量，比 cosine 与相对 L2；再比两者过 MLP 后的 `(E,ν,ρ)` | **04**（它手上同时有 feat_map 和 gau_phys_feat，是唯一天然的位置） |
| 2 | **成员集合**：GT mask 反投影的高斯 vs HDBSCAN 聚的高斯 | 在 bench 上同时用两种成员集合过同一个 MLP，比两组 (E,ν,ρ) | **06**（新开，见下）——它要训好的头，02 和 04 都得先落地 |

⚠️ 两处都**不许用 `torch.equal`**（地图 F4：trace kernel 用 `atomicAdd`，
同一条路跑两遍差 maxabs 3.4e-3 / 相对 1.7e-6）。比较要带控制组：
同一条路跑两遍的差是噪声底，别的差要显著高于它才算数。

### 给 02 的交待（这是本票对关键路径的全部输出）

**训练侧照原样配，不改头、不改 loss、不改 `instance_mask` 的来源。**
02 只做地图 G6 说的那件事——把 dataset 从 scannet100 换成 Infinigen 物性语料，
定死运行点，起跑。`physgm_dpt` / `phys_feat_dim: 32` / `physgm_hidden: 64` 全部保持。

### 给 04 的交待

trace 两路特征落同一份高斯；**必须一并落盘 `num_ray`**（否则 (a) 的加权无法在事后做，
要重跑 3 趟 trace）；并完成上表第 1 行的测量。

### 被本票排除的

- **候选 B / C 判出本图 scope**（不是"以后不做"，是"不在这两天做"）：
  B 把 HDBSCAN 塞进训练循环，C 要改头 + 改 loss + 重训。
  两者都在关键路径上加一个未知量，而 A 的代价是可量的。已写进地图 Out of scope。

# 06 — 量「成员集合」这处不等价：GT mask 的高斯 vs HDBSCAN 的高斯

Type: task
Status: open
Blocked by: [02 — 物性头训练接线与运行点](02-train-physics-head.md) ✅ closed, [04 — trace 两路特征](04-trace-two-feature-sources.md) ✅ closed
Blocks: —
Assignee: —

> **诊断票，不阻塞交付。** 01 号票判了候选 A，A 不消除训练/推理失配，
> 它把失配挪到**可量的位置**。两处不等价里的第一处（池化口径）在 04 量；
> 第二处（成员集合）要训好的头才能量，落在这里。
> 排在 05 之后做；时间不够就如实标「未量」，**不许用「看起来差不多」代替**。

## Question

`physgm_dpt` 训练时的实例成员是 **GT instance mask 圈的像素**；
推理时是 **HDBSCAN 在高斯上聚出来的高斯集合**。这条差异不可能消除，只能量。

**做法**：在 `3dovs/bench` 上，用**同一份 `gau_phys_feat`、同一个训练好的 MLP**，
只换成员集合，比两组 (E, ν, ρ)：

- **甲**：GT mask 反投影到高斯 → 该集合上 `num_ray` 加权平均 → MLP → denormalize；
- **乙**：HDBSCAN 实例的高斯集合 → 同样的平均 → 同一个 MLP → denormalize。

⚠️ 甲乙的实例**不是一一对应的**（HDBSCAN 会把一个 GT 物体切开、也会把几个并起来）。
先定一条配对规则（建议：高斯集合的 IoU 最大者配对，并报未配上的两侧数量），
配对规则本身要写进 Answer——不同的配对规则会给出不同的差值。

## 一并要报的

- **噪声底**：同一条路跑两遍的差（地图 F4：`atomicAdd`，maxabs 3.4e-3 / 相对 1.7e-6）。
  差值不显著高于噪声底就不算差异。⚠️ 不许用 `torch.equal`。
- **`mu_spread`**（01 号票 (d) 条的定义）：每个高斯单独过一遍同一个 MLP 得到的 `mu` 的标准差。
  甲乙两侧都报——若乙的 `mu_spread` 明显大于甲，说明 HDBSCAN 把不同材质的东西聚进了一个实例，
  这比 (E,ν,ρ) 的均值差更能说明问题。
- **头输出的 `var`** 与 `mu_spread` **分开摆**，别合成一个数。

## 完成判据

一张表：每个配上的实例一行，甲的 (E,ν,ρ)、乙的 (E,ν,ρ)、相对差、两侧的 `var` 与 `mu_spread`；
加上噪声底、配对规则、未配上的实例数。

一句话结论：**成员集合这处不等价，在本图的判据（「有个值就行」）下是可忽略的，还是不可忽略的。**
若不可忽略，写清楚它会怎么影响后续（那是下一张图的事，不是本图的）。

---

## 04 号票交下来的前提（2026-09-09，别再重新发现）

1. **甲（GT mask 侧）的高斯成员集合 04 已经算好了**：把 mask 的 one-hot **加一张常数 1.0
   覆盖平面**穿过同一批相机同一份高斯，`share = occ/coverage`，`argmax` 且 `share > 0.5`。
   1,005,708 个被 trace 的高斯里 512,699 个落到某实例，其余是 GT 留 0 的墙/地板。
   **照抄 `scripts/check_iggt_phys_trace.py`，不要另发明一套反投影。**
2. **bench 的 GT 是 10 个实例**（36 视角并集上 id 1…10），不是老图记的 8 个。
3. **池化就是「把该实例的高斯的 `gau_sem` 加起来」**。分母（`Σ num_ray` / `Σ blend_mass`
   / 任何正标量）因 decoder 首层 LayerNorm 的尺度不变性**完全不影响 MLP 的输入**。
   别在分母上花时间。
4. **预期主因是实例尺寸，不是加权**：04 实测 2D↔3D cos 在 >1000 高斯的实例上 0.95–0.99，
   在 534–1306 高斯的四个小/薄实例上塌到 0.39–0.82；加权 vs 等权 cos = 0.996。
   ⇒ 本票的表**必须带 `n_gau` 列并按它排序**，否则会把尺寸效应误读成成员集合效应。
5. **噪声底 ≈ 6e-7 相对**（04 的控制组读数）。
6. ⚠️ `mu_spread` 逐高斯 decode 时，喂 MLP 的向量比训练时短约 50×、逐高斯尺度散布 100×，
   **只因为 LayerNorm 尺度不变才有意义**。若发现 `mu_spread` 反常，先怀疑这条。
7. 现成资产：`.pt` 在
   `/mnt/storage_pool/liaoyuanjun/runs/ticket04_iggt_phys/bench/gaussian_iggt_phys_feat.pt`；
   harness 在 `scripts/check_iggt_phys_trace.py`（docstring 里有两条命令，约 80 s 复现）。

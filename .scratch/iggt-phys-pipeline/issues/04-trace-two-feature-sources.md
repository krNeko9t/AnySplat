# 04 — trace 侧同时搬 instance feat 和 physics feat

Type: task
Status: open
Blocked by: [01 — 物性怎么从 dense feat 走到 3D 实例](01-physics-from-dense-feat-to-3d-instances.md) ✅ closed
Blocks: [05 — 交付物](05-deliverables.md), [06 — 量两处不等价](06-quantify-the-two-mismatches.md)
Assignee: —

> 01 号票判了**候选 A：3D 池化，头不动**。本票就是把那条路实现出来。
> **可以和 02 的训练并行**（trace 的是特征，物性 MLP 是最后一步才用的）。

## Question

让一次 trace 同时把两路特征落到**同一份高斯**上：

- **instance feat**：IGGT 的 8 维（`instance_feat_dim: 8`）→ HDBSCAN → 实例 id；
- **physics dense feat**：`physgm_dpt` 的 DPT 输出，32 维（`phys_feat_dim: 32`），
  image 分辨率 → 每个实例在 3D 里平均 → 过训练好的 per-property MLP。

### 已知的形状与成本（G5）

`TRACE_CHANNELS = 20`（`src/trace_render/trace_rasterize.py:20`）
⇒ 8 维 1 趟 + 32 维 2 趟 = **3 趟**。
老图 02 号票做的 `trace_single_view_chunked` 就是为这件事准备的（168.5 ms/view
⇒ bench 36 帧约 6 秒，garden 185 视角约 31 秒）。**不重编译 `TRACE_CHANNELS`。**

### 要做的

1. 把 IGGT 的 physics dense feat **接成 trace 脚本的一个 feature source**
   （`scripts/trace_instance_to_gaussians.py` 本来就是多源的；
   老图 03 号票给 SegVGGT 加源时抽出的 `prep` 契约可以照抄形状）。
2. 两路特征落进**同一个 `.pt`**，且**在同一份高斯、同一批相机上**——
   这是「同一份高斯」这句话的全部内容，要有一条断言挡住它。
3. ⚠️ **`num_ray` 必须一并落盘。** 01 号票定的池化口径是
   **按 `num_ray` 加权**（因为训练侧是「跨视角逐像素等权」，
   `physics_pool.py:79-80` 的 `masked.sum(dim=(0,2)) / count`，
   而 trace 给出的是 `feat_g = sum_gau_sem_g / num_ray_g`
   ⇒ `Σ num_ray·feat / Σ num_ray` 才等于像素均值）。
   不落 `num_ray`，事后想加权就要重跑 3 趟 trace。
4. ⚠️ **判据不能用 `torch.equal`**（地图 F4：`atomicAdd`，同一条路跑两遍差 3.4e-3，
   相对 1.7e-6）。比较要带控制组，参照 `scripts/check_chunked_trace.py` 的写法。

### 5. 量「不等价 1」：池化口径（01 号票分派到本票）

本票是**唯一同时握着 2D `feat_map` 和 3D `gau_phys_feat` 的位置**，所以这条在这里量：
同一张 mask 下，
- **2D 口径**：`pool_one_sample` 在 image 分辨率上逐像素等权池出的 32 维向量（= 训练口径）；
- **3D 口径**：同一实例的高斯集合上，`num_ray` 加权 / 等权两版 32 维向量。

报 cosine 与相对 L2（三对：2D↔3D加权、2D↔3D等权、加权↔等权），
并把噪声底（同一条路跑两遍）一起报。**MLP 之后的 (E,ν,ρ) 差留给 06**（那时头才训好）。

## 完成判据

一个 `.pt`，里面有 `gau_inst_feat [P,8]`、`gau_phys_feat [P,32]`、**`num_ray [P]`**、
以及它们共享的高斯索引；被 trace 到的高斯占比报出来（老图在 bench 上是 96.2%，可作参照）；
带控制组的一致性读数；以及上面第 5 条的池化口径读数。**MLP 解码留给 05。**

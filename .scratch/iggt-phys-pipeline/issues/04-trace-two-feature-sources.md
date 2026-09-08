# 04 — trace 侧同时搬 instance feat 和 physics feat

Type: task
Status: closed
Blocked by: [01 — 物性怎么从 dense feat 走到 3D 实例](01-physics-from-dense-feat-to-3d-instances.md) ✅ closed
Blocks: [05 — 交付物](05-deliverables.md), [06 — 量两处不等价](06-quantify-the-two-mismatches.md)
Assignee: subagent (2026-09-09)

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

---

## Answer（2026-09-09 01:00 关闭）

### 产物

`/mnt/storage_pool/liaoyuanjun/runs/ticket04_iggt_phys/bench/gaussian_iggt_phys_feat.pt`
**208 MB**，`3dovs/bench`，36 视角，**N = 1,045,236 高斯**，surfel backend。

| key | shape | dtype |
|---|---|---|
| `gau_inst_feat` | (1045236, 8) | float32 |
| `gau_phys_feat` | (1045236, 32) | float32 |
| `num_ray` | (1045236,) | float32 |
| **`blend_mass`** | (1045236,) | float32 |
| `gaussian_index` | (1005708,) | int64 |
| `valid_mask` | (1045236,) | bool |
| `feat` / `feat_unnorm` | (1045236, 8) | float32 |

所有逐高斯数组都是全长 N、共享同一套下标；`gaussian_index` 是被 trace 到的那 1,005,708 个
（压缩视图 = `t[gaussian_index]`）。**覆盖率 1,005,708 / 1,045,236 = 96.22%**，
与老图 bench 的 96.2% 对上。
旁产物：`bench_run2/`（控制组第二次跑）、`bench_iggt_only/`（旧默认路径）、
`bench/feat_maps/`（466 MB 逐视角 fp16 2D 图）、`physhead_step500.pt`、`check_report.txt`。

### 墙钟

3 趟（8 维 ×1 + 32 维 ×2）36 视角 **2.85 s = 79.2 ms/view**；带 `--trace_blend_mass` 4 趟
3.81 s = 105.9 ms/view。单趟 `iggt` 对照 23.8 ms/view，与老图记的 23.5 ms 对上，标度线性。
端到端 37.6 s（含模型加载与 5 s IGGT 前馈）。**没重编译 `TRACE_CHANNELS`。**

### 改动与「加性」证明

- `scripts/trace_instance_to_gaussians.py`：+362 / **−2**。删掉的两行只是
  `--feature_source` 的 `choices` 列表（追加一项）和一句 print（追加后缀）。
  其余全新增：`prepare_iggt_phys_features`、`FEATURE_PREP["iggt_phys"]`、
  `prep` 契约的可选 `aux_feat_maps`/`primary_name`/`provenance` 扩展（照抄老图 03 给
  `query_bank` 用的形状）、三个默认关闭的 flag。
- `scripts/check_iggt_phys_trace.py`：新增，是下面所有读数的可复现 harness。
- **没碰** `src/`、`config/`、`instseg_infer.py`、`hdbscan_assign.py`。

**旧默认路径不变的数值证明**（同 flag 跑 `--feature_source iggt` 与 `iggt_phys` 的实例流对比）：

| 比较 | maxabs (裸 `gau_sem`) | mean | rel |
|---|---|---|---|
| `iggt_phys` vs `iggt` | 1.953e-02 | 4.762e-07 | 2.581e-07 |
| **控制组**（同一条路两遍） | **1.953e-02** | 4.664e-07 | **2.581e-07** |

三位有效数字与控制组相同，`num_ray` 逐位相等，被 trace 的高斯同为 1,005,708。
`pytest tests/` 71/71 过。

### 控制组读数（地图 F4：不许 `torch.equal` 比特征）

| | maxabs | mean | rel |
|---|---|---|---|
| `gau_inst_feat` | 1.043e-07 | 1.357e-10 | 5.90e-07 |
| `gau_phys_feat` | 1.192e-07 | 1.599e-10 | 2.46e-07 |
| `num_ray` | 逐位相等 | | |

**噪声底 ≈ 6e-7 相对**。低于它的差一律不算。
唯一用 `torch.equal` 的地方是每个 aux 趟对 `num_ray` 的断言——
`num_ray` 是只依赖几何 + 相机 + `img_mask` 的整数计数（`trace_rasterize.py:255-256`），
老图已证跨趟逐位相等，**只有这里精确比较是对的**。这条断言就是「同一份高斯」的执法。

### 池化口径（「不等价 1」）

**mask**：场景自带 GT `sam/mask/*.png`，756×1008，36 视角并集上 id 1…10。
**2D 侧**在模型自己的 336×504 输出网格上池（mask 最近邻降采样）；
**3D 侧**把 mask 的 one-hot **加一张常数 1.0 覆盖平面**穿过**同一批相机、同一份高斯**
（断言 `num_ray` 相同），`share = occ/coverage`，`argmax` 且 `share > 0.5`。
1,005,708 个里 512,699 个落到某个实例上，其余是 GT 留在 0 的墙/地板。
**头权重**：02 号票 step 500 的 checkpoint（10000 步里的 500，**未收敛**），骨干是官方 IGGT。

| | cos 2D–3D加权 | cosLN | cos 2D–3D等权 | cosLN | cos 加权–等权 | ‖3Dw‖/‖2D‖ |
|---|---|---|---|---|---|---|
| **均值（10 个实例）** | **0.8725** | 0.8672 | **0.8782** | 0.8736 | **0.9961** | 0.0199 |

噪声底：3D 侧 cos = 1.000000 / relL2 ≤ 8.3e-08；2D 侧 cos = 1.000000 / relL2 ≤ 1.6e-06。
**上表每个数都高出噪声底六个数量级，失配是真的。**

三条读法：
1. **相对 L2 那一列没有信息量**（恒在 ~1.93）——3D 向量比 2D 短约 50×，
   纯幅度差下 relL2 饱和到 2.0。真正进 MLP 的是过 LayerNorm 之后的方向，故报 `cosLN`，
   它与裸 cos 几乎重合。
2. **2D↔3D 是实例尺寸效应，不是加权效应**：
   高斯数 > ~1000 的七个实例 cos ≈ 0.95–0.99；
   四个小/薄实例（534–1306 高斯）塌到 **0.39–0.82**。
3. **加权 vs 等权 cos = 0.996**，且等权在均值上还略近 2D（0.8782 vs 0.8725）。
   ⇒ **加权口径的选择在 bench 上几乎无关紧要**，它是有原则的选择，不是有效果的选择。
   06 号票该预期「实例尺寸」而不是「加权」是主因。

### ⚠️ 两条与地图/01 号票冲突的实测，已回写

**(a) `feat_g` 不是「打中该高斯的光线上的像素均值」，01 号票 (a) 的推导是错的。**
详见 [01 号票的更正块](01-physics-from-dense-feat-to-3d-instances.md)。
一句话：`gau_sem` 是 alpha 加权累积、`num_ray` 是不加权计数
（`trace_rasterize.py:251` 与 `:255-256` 白纸黑字），
常数 1.0 实测 `gau_sem/num_ray` 的 p50 = 0.0124 而非 1.0，p95/p05 = 114×。
**结论（`num_ray` 加权）不变且更强**：`Σ num_ray·feat = Σ gau_sem = Σ_r W_r f_r`，
是实例足迹上 alpha 加权的光线级像素和；而分母因 LayerNorm 的尺度不变性**完全不重要**。
为此 `.pt` 里加了 `blend_mass`（+26 ms/view），日后要显式除掉尺度不必重跑 trace。

**(b) bench 的 GT mask 是 10 个实例，不是老图记的 8 个。**
`sam/mask/*.png` 在 36 视角并集上是 id 0…10（0 是背景/stuff）；单视角更少（view 00 是 0…8）。

### 没做 / 待决

- **只跑了 bench。** garden 不在本机的老图路径下，「garden 在不在」是 03 号票的事。
- **头在 step 500/10000。** 管线是精确的、池化几何与权重无关，但具体的 32 维向量会随训练移动。
  照 `check_iggt_phys_trace.py` docstring 里的两条命令对新 checkpoint 重跑，
  **约 80 s 复现整张表**。
- ⚠️ **分辨率张力，标出未决，留给 05。**
  本票跑在 504×336（`iggt` 默认），为的是实例流与 03 号票的验收基线逐位可比；
  但物性头是在 **252×448、4 固定视角**上训的，等于在略微偏离训练分辨率的地方评估它。
  `--iggt_phys_image_size 448,252` 可切换。**05 号票要决定哪一头的一致性更重要。**
- MLP 解码、`(E,ν,ρ)`、`mu_spread`、GT-vs-HDBSCAN 成员对比 → 05 / 06。

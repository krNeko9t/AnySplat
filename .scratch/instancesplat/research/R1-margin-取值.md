# R1 — 实例 margin 的取值（δ_pull / δ_push / δ_cross）

> 票：`.scratch/instancesplat/tickets/R1-margin-取值.md`
> 结论分三档标注：**[论文]** = 一次来源里白纸黑字；**[代码]** = 从官方/本仓库源码读到；**[推理]** = 我从前两者推导，未经实验验证。

## 0. 三个数值（先给结论）

| 量 | 取值 | 一句话依据 | 档位 |
|---|---|---|---|
| **δ_pull** | **0.2** | De Brabandere「δ_d > 2δ_v」换算到 InstanceSplat 的量纲 = δ_pull < δ_push/4 = 0.25，取 0.2 留余量 | [推理]，约束来自 [论文] |
| **δ_push** | **1.0** | IGGT 在**同样是 ℓ2 归一化的 8 维实例特征**上用 M=1.0，且 λ_pull/λ_push=2/1 与 InstanceSplat 完全一致 | [论文]（IGGT 附录 A.3） |
| **δ_cross** | **0.3** | 必须 > δ_pull（跨视角噪声源更多），必须 < 2δ_pull（否则不再约束任何东西），取区间中点 | [推理] |

不确定性：**δ_push=1.0 的置信度最高**（有直接可比的一次来源 + 独立几何论证收敛到同一个数）。
δ_pull / δ_cross **没有任何一次来源给过数值**，是从约束区间里取的点，建议区间见 §5。

---

## 1. InstanceSplat 自己写了什么、没写什么

**[论文]** 4.1 Implementation Details 把 λ 全给了，**δ 一个没给**：
> "We set λ_p = 0.05, λ_ins = 0.01, λ_sem = λ_bd = 0.02, and (λ_pull, λ_push, λ_cross) = (2, 1, 2)"
— `ref_knowledge/InstanceSplat/InstanceSplat.md:194`

正文只说三个 δ 是 "margin hyperparameters"（`InstanceSplat.md:85`）。全文无 δ 的消融、无附录数值。**这是真空白，不是我没找到。**

### 1.1 量纲：是欧氏距离，不是余弦 **[论文]**

Eq.5 / Eq.6 / Eq.7 三处全部写成 `‖·‖₂`（`InstanceSplat.md:88 / :94 / :105`）：

- Eq.5 `L_pull = E_p [ ‖f_i(p) − f̄_k^(i)‖₂ − δ_pull ]_+`
- Eq.6 `L_push = E_{k<l} η_{k,l} [ δ_push − ‖f̄_k^(i) − f̄_l^(i)‖₂ ]_+`
- Eq.7 `L_cross = E_{i<j} [ ‖f̄_k^(i) − f̄_k^(j)‖₂ − δ_cross ]_+`

余弦只出现在**边界分支** Eq.9（`InstanceSplat.md:122`，`r_i(p) = max(1 − z_i(p)ᵀz_i(q))`）。
→ **三个 δ 的量纲是「单位球上的弦长」，取值范围 [0, 2]。**

### 1.2 三个必须注意的形式差异 **[论文]**

**(a) Eq.6 的 margin 没有乘 2。** De Brabandere 的 push 是 `[2δ_d − ‖μ_A−μ_B‖]_+`，InstanceSplat Eq.6 是 `[δ_push − ‖f̄_k−f̄_l‖]_+`。
→ **δ_push 直接就是目标原型间距**，而 De Brabandere 的目标间距是 `2δ_d = 3.0`。换算时必须除以 2，否则差一倍。

**(b) hinge 没有平方。** De Brabandere 三项全是 `[·]_+²`（见 §2），InstanceSplat Eq.5/6/7 都是一次方。
→ 梯度在 hinge 内是常数而非线性衰减，靠近 margin 时压强更大，**这也是 δ 不宜取大的一个理由**。

**(c) 原型不再归一化 —— 容易漏，影响 δ_push 的可达上界。**
Eq.4：`f̄_k^(i) = (1/|Ω|) Σ_p f_i(p)` —— **外面没有 norm(·)**（`InstanceSplat.md:80`）。
对比语义分支 Eq.14：`ℓ̄_k^(i) = norm( E_p[ norm(E_i(p)) ] )` —— **有**外层 norm（`InstanceSplat.md:157`）。
这个不对称是论文里显式写出来的。→ **像素特征在单位球面上，原型在单位球内部**，`‖f̄_k‖ = c_k ≤ 1` 恰好是该实例的紧致度。
后果：两原型的最大可达距离 ≈ `c_k + c_l ≤ 2`，实际（c≈0.95）约 1.9，**比球面上限还紧**。

---

## 2. 原始论文的尺度：De Brabandere et al. 2017

一次来源：arXiv:1708.02551 全文（ar5iv 渲染版 <https://ar5iv.labs.arxiv.org/html/1708.02551>）。

**[论文]** 损失形式（注意平方 hinge 与 `2δ_d`）：
- `L_var  = (1/C) Σ_c (1/N_c) Σ_i [ ‖μ_c − x_i‖ − δ_v ]_+²`
- `L_dist = (1/(C(C−1))) Σ_{c_A≠c_B} [ 2δ_d − ‖μ_{c_A} − μ_{c_B}‖ ]_+²`

**[论文]** 数值：**δ_v = 0.5，δ_d = 1.5**，CVPPP 与 Cityscapes 两个实验都用这一组。
**[论文]** 维度：Cityscapes **8 维**、CVPPP 16 维。**全文未提 ℓ2 归一化 —— 嵌入是无约束的。**

**[论文]** 两条 margin 关系（3.2 Post-processing，原文措辞）：
1. "all cluster centers are at least 2δ_d apart"
2. "If δ_d > δ_v, then each embedding is closer to its own cluster center than to any other cluster center."
3. "If we set δ_d > 2δ_v, then each embedding is closer to all embeddings of its own cluster than to any embedding of a different cluster."

→ 常被转述成「δ_d > 2δ_v」的那条，是**更强**的那个（保证任意两个嵌入之间的可分，而不只是到中心可分）。这条才是 HDBSCAN 能干活的条件。

**本仓库现状 [代码]**：`config/loss/disc.yaml:3-4` 正是 `delta_v: 0.5` / `delta_d: 1.5`，
`src/loss/loss_disc.py:148` 用 `2 * delta_d - off_diag`（与原论文一致），
`src/loss/loss_disc.py:130` 用平方 hinge（一致），
**`src/loss/loss_disc.py:109` 的 `emb_flat = F.normalize(...)` 是注释掉的** —— 确认这条线跑在**未归一化**嵌入上，票里的判断正确。

### 2.1 为什么 (0.5, 1.5) 搬不过来 **[推理]**

把 δ_d=1.5 塞进 Eq.6 的位置，目标间距是 `2δ_d = 3.0`，而 **归一化后两点最大距离是 2**。
→ push 的 hinge `[3.0 − d]_+ ≥ 1.0` **永远打不开**，变成一个恒正的常数惩罚项，梯度方向永远是「继续推」，
push 项失去「够开了就松手」的语义，直接和 pull 打架。这是最硬的反证。
即便按不乘 2 的 Eq.6 读成 δ_push=1.5，也见 §3.2：8 维下超过 ~16 个实例就几何上不可能。

---

## 3. 归一化之后的几何论证

设像素特征 `f = norm(S̃)` 在单位球面上，两单位向量夹角 θ 时弦长 `d = 2·sin(θ/2)`：

| θ | 30° | 60° | 90°（正交） | 120° | 180°（反向） |
|---|---|---|---|---|---|
| d | 0.518 | **1.000** | **1.414 (√2)** | 1.732 | **2.000** |

### 3.1 硬上界 **[推理]**
`δ_push < 2` 是绝对上界（否则 hinge 永不闭合，同 §2.1）。考虑原型在球**内**（§1.2c），
实际可达上界 ≈ `2c ≈ 1.9`。取到 1.9 附近意味着要求所有实例两两接近反向 —— 8 维里最多容纳 **9** 个。

### 3.2 容量上界：δ_push 决定一个视角里最多能放几个实例 —— 这是决定性的约束

8 维特征，一个室内视角常见 10–40 个实例。问「K 个单位向量能否两两距离 ≥ δ_push」：

- **δ_push = √2 ≈ 1.414（正交）**：等价于要求两两内积 ≤ 0。
  **[论文/数学定理]** ℝ^d 中两两内积非正的非零向量最多 **2d** 个（严格正交时最多 d 个）
  → 8 维最多 **16** 个（严格正交只有 8 个）。<https://siongui.github.io/2017/05/21/maximum-number-of-pairwise-non-acute-vectors-in-R-n/>
  → **一个视角只要超过 16 个实例，δ_push=√2 就是数学上不可满足的**，push 项永久饱和。**排除。**

- **δ_push = 1.0（60°）**：等价于 kissing number 问题。
  **[论文/数学定理]** 8 维 kissing number = **240**（E8 格，Odlyzko–Sloane / Levenshtein 1979，Annals 168(2008) 复核）
  <https://annals.math.princeton.edu/wp-content/uploads/annals-v168-n1-p01.pdf>
  → **8 维里可以放 240 个两两距离 ≥ 1.0 的单位向量**，远超任何真实视角的实例数。**充分宽裕。**

**结论 [推理]：δ_push = 1.0 恰好是 8 维下「容量最大化」的那个点** —— 再大一点容量断崖式下跌（1.0→240，1.414→16），
再小一点则白白浪费可分性预算。这个几何论证**独立地**收敛到 IGGT 的 M=1.0（§4），是本票置信度最高的一条。

（注：原型模长 c≈0.95 时，要达到 d=1.0 需夹角 ≈63.5°，容量略降但仍在百量级；
 而 d=1.414 需夹角 ≈96°，比正交还苛刻，进一步坐实排除。）

### 3.3 δ_pull 的上界：把 De Brabandere 的约束换算过来 **[推理]**

记簇半径 `r = δ_pull`、目标间距 `S = δ_push`（Eq.6 不乘 2，§1.2a）。
复刻 De Brabandere 的强条件推导：最大簇内距 `2r`，最小簇间距 `S − 2r`，要求 `2r < S − 2r`：

> **δ_pull < δ_push / 4 = 0.25**

（对照：De Brabandere 自己 r=0.5、S=2δ_d=3.0，`r/S = 1/6`；等比例缩到 S=1.0 是 **0.167**。）
弱条件（只保证靠自己中心更近）是 `δ_pull < δ_push/2 = 0.5`，不够 HDBSCAN 用。

→ **δ_pull 的安全区是 (0, 0.25)**，De Brabandere 的等比例点是 0.167，我取 **0.2**（角锥半径 ≈ 11.5°，
够吸收实例内的光照/纹理变化，又离 0.25 天花板有余量）。

---

## 4. 近邻工作实测值对照表

| 工作 | 特征是否 ℓ2 归一化 | 目标形式 | margin 实际值 | 维度 | 可比性 |
|---|---|---|---|---|---|
| **De Brabandere 2017** | **否** | 平方 hinge，push 用 `2δ_d` | δ_v=0.5, δ_d=1.5（间距目标 3.0） | 8 / 16 | 形式同源，**尺度不可搬** |
| **IGGT (ICLR'26)** | **是** | `max(0, M − d)`，d = 归一化特征的 L2 距离 | **M = 1.0**，λ_pull=2.0, λ_push=1.0 | **8** | **完全可比** ✅ |
| **本仓库 `loss_mvc.py`** | **是** | `relu(margin − d)` | **margin = 1.0**, λ_pull=2.0, λ_push=1.0 | — | **完全可比**（就是 IGGT 那套）✅ |
| 本仓库 `loss_disc.py` | 否（`:109` 注释掉） | De Brabandere 原式 | 0.5 / 1.5 | — | 未归一化，不可比 |
| Gaussian Grouping (ECCV'24) | N/A（走分类器） | 交叉熵 + 3D KL 正则 | **无 margin** | 16 | 范式不同，无参考价值 |
| Contrastive Lift (NeurIPS'23) | **否** | `exp(−d²/temperature)` softmax | **无 margin**，只有 temperature | — | 范式不同 |
| OpenGaussian (NeurIPS'24) | **否** | 逆距离排斥 `1/(d²+ε)`，ε=1 | **无显式 margin** | 6 | 范式不同 |

### 4.1 IGGT 是决定性证据 **[论文]**

一次来源：IGGT 论文 <https://arxiv.org/html/2510.22706v1>（v1 与 v3 两版独立核对，数值一致）。

- 损失就叫 **ℒ_mvc**（与本仓库 `src/loss/loss_mvc.py` 同名，非巧合）：
  `ℒ_mvc = λ_pull · Σ d(f_pi, f_pj) + λ_push · Σ max(0, M − d(f_pi, f_pj))`
- **"d(·,·) is the L2 distance between normalized features"** —— 与 InstanceSplat Eq.5/6 完全同构。
- 附录 A.3 Training Details：**"we set λ_pull = 2.0, λ_pull = 1.0 and M = 1.0"**
  （原文第二个 `λ_pull` 是笔误，按公式应为 λ_push；v1/v3 两版都带这个笔误）
- 实例特征 **8 维**："map them through a conventional 3×3 convolutional layer to 8 dimensional instance features O_ins ∈ ℝ^{N×8×H×W}"

**三处对齐 [推理]**：InstanceSplat 的 (λ_pull, λ_push) = **(2, 1)** 与 IGGT 的 (2.0, 1.0) **逐位相同**，
实例维度同为 **8**，距离定义同为「归一化特征的 L2」。→ 强烈提示 InstanceSplat 的实例分支就是照 IGGT 的配方改的，
**δ_push 继承 M = 1.0 是最小假设**。

⚠️ 证据边界：`lifuguan/IGGT_official` 仓库**只放了推理代码**（`demo.py` + `sam2/`，无 loss、无训练 config，
经 GitHub tree API 全量列举确认）。所以 M=1.0 是**论文附录写的**，不是从官方代码读到的。

### 4.2 本仓库已有的旁证 **[代码]**

`src/loss/loss_mvc.py` 就是这套配方的实现，且**已经在归一化特征上跑**：
- `:435` `f = F.normalize(f, p=2, dim=-1, eps=1e-8)`
- `:447-448` `dist2 = (2.0 - 2.0*dot).clamp_min(0.0)` → `dist = sqrt(dist2)` ← 明确的球面弦长，值域 [0,2]
- `:457` pull = `dist.sum()`（**无 margin，纯距离**）
- `:458` push = `F.relu(margin - dist)`
- `config/loss/mvc.yaml:4-6` → `margin: 1.0`、`lambda_pull: 2.0`、`lambda_push: 1.0`

→ **δ_push = 1.0 在本仓库的归一化路径上已经是既定值**，不是新引入的超参。

⚠️ **但注意**：IGGT / `loss_mvc.py` 的 pull 项是**纯距离、没有死区**（等价 δ_pull = 0）。
InstanceSplat Eq.5 显式引入了 δ_pull —— 这是 InstanceSplat 相对 IGGT 的**增量**，
所以 δ_pull **没有任何一次来源可继承**，只能按 §3.3 的约束推。**这是本票最大的不确定点。**

---

## 5. δ_cross：应该比 δ_pull 大 **[推理]**

### 5.1 下界 δ_cross > δ_pull —— 噪声源更多
δ_pull 要吸收的是：**同一视角、同一物体、同一次渲染**内像素之间的散布。
δ_cross 要吸收的是上面全部，**再加**三项 δ_pull 完全不承担的：
1. **可见集不同**：`Ω_k^(i)` 与 `Ω_k^(j)` 是物体的**不同部分**（遮挡、截断、视角）。极端情况两者几乎不相交，
   两个原型是**两个不同子总体的均值**，而不是同一总体的两次采样。
2. **视角相关外观**：高光、光照方向变化会渗进渲染出的实例特征。
3. **alpha 合成权重不同**：同一批高斯在两个视角下的合成权重不同，即使 3D 特征完全一致，渲染结果也有残差。

把 δ_cross 设得比 δ_pull 还紧，等于**对更嘈杂的量用更严的尺**：
即使实例学得很好，L_cross 也会长期非零，而 λ_cross = 2（与 λ_pull 同为最大权重，`InstanceSplat.md:194`），
这份持续梯度会直接打进**共享的 3D 高斯特征**上 —— 那正是唯一能让训练失稳的地方。

### 5.2 上界 δ_cross < 2·δ_pull —— 再大就不约束任何东西
两个视角的原型都是「半径 δ_pull 的实例特征球」内某个子集的均值，**最坏情况相距 2δ_pull**。
若 `δ_cross ≥ 2δ_pull`，则两个原型即使落在实例自身散布的**两个对立端**也零损失
—— L_cross 退化成恒零项，跨视角一致性这条线白搭。

### 5.3 取值
`δ_pull < δ_cross < 2δ_pull` = **(0.2, 0.4)**，取中点 **δ_cross = 0.3**。

**交叉校验（簇可分性预算）[推理]**：跨视角合并后单个实例簇的有效半径
`R ≈ sqrt(δ_pull² + (δ_cross/2)²) = sqrt(0.04 + 0.0225) = 0.25`（高维近似正交，按平方和而非线性相加）。
两簇直径和 `2R = 0.5`，对上 `δ_push = 1.0` → **间距/簇径 = 2:1**，正是密度聚类（HDBSCAN）舒服的比例。
最坏情况（完全共线）`2δ_pull + δ_cross = 0.7 < 1.0`，仍不重叠。
→ (0.2, 1.0, 0.3) 这一组在两种口径下都自洽。

---

## 6. 调参方向：早期 loss 塌到 0 怎么办

先说**健康的初始状态长什么样 [推理]**（这决定了「塌 0」是不是真异常）：
初始化时实例头是新的，像素特征近随机 → 归一化后近似均匀分布在球面上 →
**原型是它们的均值，模长 ≈ 1/√|Ω| ≈ 0**（因为 Eq.4 不再归一化，§1.2c）→
所有原型挤在原点附近 → `‖f̄_k − f̄_l‖ ≈ 0` → **push ≈ δ_push = 1.0（满值）**，
`‖f_i(p) − f̄_k‖ ≈ 1` → **pull ≈ 1 − δ_pull = 0.8（接近满值）**。
→ **三项在 step 0 都应该显著非零。任何一项 step 0 就是 0，都值得怀疑。**

分情况：

| 现象 | 诊断 | 调法 |
|---|---|---|
| **只有 push 早早塌 0**，pull/cross 还高 | 原型间距轻易超过 δ_push，可分性预算给少了 | **调大 δ_push**：1.0 → 1.2。**硬上限 1.41**（§3.2：超过就是 16 实例的几何天花板） |
| **只有 pull（和/或 cross）塌 0** | 死区太宽，特征躲进死区偷懒，原型仍然糊 | **调小死区**：δ_pull 0.2 → 0.1，δ_cross 0.3 → 0.15。**不要靠调大 δ_push 补救** |
| **三项同时塌 0** | 几乎肯定不是 margin 问题 | 查管线：mask 是否全空、`ignore_id=0` 是否吞掉了全部像素、`|K_i| ≥ 2` 是否成立（只有 1 个实例时 push 无配对天然为 0） |
| **pull/cross 塌 0 且原型互相靠得很近**（塌缩解） | 全部实例映射到同一方向的平凡解 | 这是 λ_pull+λ_cross=4 vs λ_push=1 造成的早期盆地。**调小死区**，必要时临时把 λ_push 提到 2；**不要调大 δ_push** |

**关键判据（免聚类，配合 T5 看板）[推理]**：只看 loss 值分不清「学好了」和「塌缩了」，
必须同时监控 **原型模长均值 `mean_k ‖f̄_k‖`** 与 **原型两两距离均值**。
- 健康：`‖f̄_k‖` 从 ~0 升到 0.9+，且原型两两距离升到 ≈ δ_push 附近并**停住**。
- 塌缩：`‖f̄_k‖` 升到 0.9+，但原型两两距离**也**趋近 0。

**一句话版本**：早期塌 0 先分是哪一项 —— **push 塌就调大 δ_push（1.0→1.2，上限 1.41）；
pull/cross 塌就调小死区（δ_pull→0.1、δ_cross→0.15）；三项同时塌就别动 margin，去查 mask / ignore_id / |K_i|。**

---

## 7. 不确定性与建议区间

| 量 | 推荐 | 建议区间 | 置信度 | 理由 |
|---|---|---|---|---|
| δ_push | **1.0** | [0.9, 1.2] | **高** | 一次来源（IGGT M=1.0，同维度/同归一化/同 λ）+ 独立几何论证（8 维 kissing number 240）双重收敛 |
| δ_pull | **0.2** | [0.1, 0.25] | **中** | 无一次来源。上界 0.25 是 De Brabandere 强条件的换算 [论文→推理]；0.2 是区间内取点。IGGT 用 0（无死区）说明偏小是安全方向 |
| δ_cross | **0.3** | [0.2, 0.4] | **中低** | 完全靠推理。区间端点 = (δ_pull, 2δ_pull)，两端都有明确失效机制，中点无更强依据 |

**不是编造精确值的地方**：δ_pull 和 δ_cross 没有任何论文或代码给过数字。
我给的是**约束区间内的一个自洽取点**，且区间端点各有明确的失效机制（见 §3.3 / §5.1 / §5.2）。
若 T9 smoke run 的曲线与 §6 的健康形态不符，按 §6 的表格调整，**区间内调，不要出界**。

**是否偏离论文**：三个 δ 论文没给，谈不上偏离；δ_push=1.0 是从最近邻工作（IGGT，同一批作者圈的配方）
继承的**已有超参**，不是新引入的。δ_pull / δ_cross 是论文**已经命名但未赋值**的超参，必须赋值才能跑，
赋值依据已在 §3.3 / §5 写明。**没有引入论文之外的任何新超参。**

---

## 来源清单

**一次来源**
- InstanceSplat 原文：`ref_knowledge/InstanceSplat/InstanceSplat.md`（Eq.4 `:80`、Eq.5 `:88`、Eq.6 `:94`、Eq.7 `:105`、Eq.9 `:122`、Eq.14 `:157`、4.1 超参 `:194`、margin 措辞 `:85`）
- De Brabandere et al. 2017, arXiv:1708.02551 全文 — <https://ar5iv.labs.arxiv.org/html/1708.02551>（abs: <https://arxiv.org/abs/1708.02551>）
- IGGT, arXiv:2510.22706 — <https://arxiv.org/html/2510.22706v1> 与 <https://arxiv.org/html/2510.22706v3>（附录 A.3 训练细节；两版核对一致）
- IGGT 官方仓库 — <https://github.com/lifuguan/IGGT_official>（经 tree API 全量列举：**仅推理代码，无 loss / 训练 config**）
- OpenGaussian 官方代码 — <https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/train.py>（`:124-138` intra-mask、`:140-169` inter-mask，`epsilon = 1`，无归一化、无 hinge margin）
- Contrastive Lift 官方代码 — <https://raw.githubusercontent.com/yashbhalgat/Contrastive-Lift/main/model/loss/loss.py>（`exp(-distance_sq/temperature)`，无归一化、无 margin）
- Gaussian Grouping — <https://arxiv.org/pdf/2312.00732> / <https://github.com/lkeab/gaussian-grouping>（16 维 Identity Encoding + 交叉熵 + 3D KL；λ_2d=1.0, λ_3d=2.0；无 margin）

**数学定理**
- kissing number k(8) = 240 — <https://annals.math.princeton.edu/wp-content/uploads/annals-v168-n1-p01.pdf>
- ℝ^n 中两两内积非正的向量最多 2n 个 — <https://siongui.github.io/2017/05/21/maximum-number-of-pairwise-non-acute-vectors-in-R-n/>

**本仓库代码**
- `config/loss/disc.yaml:3-4`、`src/loss/loss_disc.py:109`（归一化被注释掉）`:130`（平方 hinge）`:148`（`2*delta_d`）
- `config/loss/mvc.yaml:4-6`、`src/loss/loss_mvc.py:435`（归一化）`:447-448`（弦长）`:457`（pull 无 margin）`:458`（push hinge）

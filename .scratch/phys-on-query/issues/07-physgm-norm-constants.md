# 07 — z-score 常数用抄来的还是本语料拟合的

Type: grilling
Status: open
Blocked by: 03

## Question

`PHYSGM_NORMALIZATION`（`src/dataset/physics/parsers.py:52-57`）的 mean/std 是**从 PhysGM 仓库
逐字抄来的**（注释自己写了 "copied verbatim"），不是本语料拟合的。[02 号票](02-labels-into-training.md)
用 `scripts/fit_physgm_norm.py` 在全部 1466 场景 / 51,981 实例上量过：

| 量 | 拟合 mean | 拟合 std | 在用 mean | 在用 std | ⇒ 实际 z 的 (均值, 标准差) |
|---|---|---|---|---|---|
| density (log10 kg/m³) | 2.8656 | 0.3993 | 3.0 | 0.5 | (−0.27, 0.80) |
| youngs_modulus (log10 Pa) | 9.4983 | 1.3206 | 7.3872 | 2.4565 | **(+0.86, 0.54)** |
| poisson_ratio (raw) | 0.3363 | 0.0663 | 0.398 | 0.111 | (−0.56, 0.60) |

**这不是无害的仿射重参数化。** 头出 `(mu, log var)` 打 Gaussian NLL：目标整体偏移 +0.86σ 且被压到
0.54 倍宽，会让 μ 的初始偏置和 σ 的量纲一起错位；三个量各偏各的，还会隐式改掉三者在总 loss 里的相对权重。

## 要决定的

1. **换成本语料拟合值**（`physgm_norm_infinigen.json`），还是**留着 PhysGM 的常数**？
   留着的唯一理由是与 PhysGM 报的数直接可比——但我们本来也不复现 PhysGM
   （memory `paper-direction-scene-phys` 明令不碰），这个理由大概率不成立。
2. 若换：**必须只在训练集上拟合**（`fit_physgm_norm.py --scene_ids`），所以阻塞于 03 定划分。
3. 若换：常数落在哪？现在是硬编码 tuple。是改死常数（简单、可复现、但换数据集就错）
   还是让 config 指一份 stats JSON（多一个失配面）。**推理期反归一化必须读同一份**，
   `physgm_denormalize` 目前也是读同一个 tuple，改一处即可，别改成两处。
4. 泊松比不取 log 且被夹在 (0, 0.5]，拟合 std 只有 0.066 —— z 会被放大到 ~1.7 倍。
   要不要对它换个变换（如 logit）而不是只换常数？

## 完成判据

常数来源定死并写进 config/代码，且 `physgm_denormalize` 与训练用的是同一套；
若选拟合，`physgm_norm_infinigen.json` 已按训练集划分重生成。

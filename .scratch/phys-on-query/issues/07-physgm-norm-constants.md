# 07 — z-score 常数用抄来的还是本语料拟合的

Type: grilling
Status: closed (2026-09-07)
Blocked by: —（原为 03；2026-09-07 阻塞已解除，且正文已被 05 号票就地执行）

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


## 2026-09-07 更新（[05 号票](05-training-operating-point.md)已就地执行）

**本票阻塞于 03 的理由消失了，而且要决定的三件事 05 号票都已执行。**

阻塞理由原本是"train-only 拟合要等 03 定划分"。05 号票为了让 val 不再跑在训练场景上，
自己加了 `scene_split_path` 并切出 **1387 训 / 74 验**（按 `scene_XXX` 生成分片整组切）
⇒ 训练集当场就有了，不必等 03。

05 号票据此执行的（提交 `40ff919` / `3a14983`）：

1. **换成本语料拟合值**。留 PhysGM 常数的唯一理由是与 PhysGM 报的数可比，
   而我们本来就不复现 PhysGM ⇒ 理由不成立。
2. **只在训练集上拟合**：`fit_physgm_norm.py --scene_ids config/experiment/splits/infinigen_phys_train_ids.txt`，
   1387 场景 / 49,214 实例。凭据 `config/experiment/splits/physgm_norm_infinigen_train.json` 进 git。
3. **落在哪：改死 `parsers.py` 的 tuple**，不走 config 指 JSON。本图只有一份语料，
   config 的灵活性买不到东西，却买来一个"推理期读了另一份"的静默错误面——
   而本票自己写了"改一处即可，别改成两处"。注释里写明拟合来源、日期、以及
   "换数据集要重新拟合，不能照抄这三行"。

| 量 | 新（训练集拟合） | 旧（抄 PhysGM） |
|---|---|---|
| density (log10 kg/m³) | 2.863740 / 0.399147 | 3.0 / 0.5 |
| youngs_modulus (log10 Pa) | 9.495947 / 1.317972 | 7.387210 / 2.456477 |
| poisson_ratio (raw) | 0.336525 / 0.066235 | 0.398 / 0.111 |

⚠️ **副作用（05 号票记在案）**：所有用 `instascene_vlm_physgm` 的配方
（`physgm_iggt` / `physgm_mvimgnet2` / `segvggt_physgm` / `segvggt_agnostic_phys_joint`）
归一化都跟着换了 ⇒ **拿旧 ckpt 做推理，反归一化对不上**。

**本票剩下的只有确认**：上面三条是不是你要的。若是，直接关票；
若第 3 条你更想要 config 指 JSON，那是一次独立的改动，在本票里做。


## Resolution（2026-09-07，本人当场追认）

**四条全部定死，票关。** 1/2/3 追认 05 号票的既成事实，4 是本票自己新答的。

### 1&2 — 换成训练集拟合值：**追认**

维持 `parsers.py:74` 的三行（density 2.863740/0.399147、E 9.495947/1.317972、
nu 0.336525/0.066235），凭据 `config/experiment/splits/physgm_norm_infinigen_train.json`
（1387 场景 / 49,214 实例，clamp 计数 27 / 23 / 0）。

**本票新查出一条票面没写的、更硬的理由**：`physics_metrics.py:125` 把 MAE 乘回 `std`
报原量纲 ⇒ **度量本身对常数选择不变**，所以"换常数是为了指标好看"这个嫌疑不成立。
但 :127 的**常数基线是 `z=0`，即"预测训练集均值"**——用旧常数时 `z=0` 是 *PhysGM 语料的*
均值，不是本语料的 ⇒ **旧常数会把主表里的常数基线做成一个人为变弱的假基线**。
而 04 号票整张票的价值正在于诚实报出 trivial baseline（论文主张 = 系统 + 度量）。
⇒ 留旧常数不只是"没好处"，是会**污染论文主表**。

### 3 — 落在哪：**维持硬编码 tuple，不走 config 指 JSON**

本图只有一份语料，config 的灵活性买不到东西，却买来"推理期读了另一份"的静默错误面。
票自己写的"改一处即可，别改成两处"就是答案。`physgm_denormalize`（`parsers.py:81`）
读同一个 tuple ⇒ 训练与推理一处定义，维持。

### 4 — 泊松比换变换（logit）：**不换**，票面前提被实测证伪

票面担心"夹在 (0,0.5]、拟合 std 只有 0.066 ⇒ z 被放大到 ~1.7 倍"。两条都不成立：

- **边界几乎没有质量**。训练划分内随机 400 场景 / 13,975 条实测：`≥0.5` 占 **0.014%**，
  `≤0` 占 0.036%。logit 要解决的"边界堆积"**在这份数据里不存在**。
- **1.7× 是相对旧的错 std 而言**。per-property 标准化本来就是要让三个量以可比尺度进 NLL，
  这不是病，正是它该干的事。

⚠️ **但实测撞见了一个票面没料到、更值得记的形状问题**：**nu 根本不是连续量，是量化网格**——
VLM 只吐圆整数，`0.35` 占 **39.7%**、`0.3` 26.6%、`0.45` 11.5%、`0.2` 5.5%、`0.4` 3.9%，
**前 5 个值吃掉 87%**。拿 Gaussian NLL 去拟合它，等于用连续密度描述一个近似分类变量。
这已经越过"常数来源"进到**改头**，按 Notes 前提第 4 条（物性头保持最朴素）押后
⇒ 转入地图 **Not yet specified**，不在本票内动。

### 4' — 旧 ckpt 反归一化对不上：**改名标记**（用户选的最轻一档）

票里只记了副作用没决定怎么办。现场核实后，受影响的 ckpt **只有一份**，不是票里担心的四份配方：

| ckpt 目录 | 用的 parser | 受影响 |
|---|---|---|
| `exp_segvggt_physgm/2026-07-29_07-35-57`（13G，60k step） | `instascene_vlm_physgm` | **是** |
| `exp_phys_prop_iggt/2026-07-14_19-18-41`（3.0G） | 另一个 parser，非 physgm | 否 |
| `exp_segvggt_finetune_agnostic/2026-07-28_21-42-50`（9.9G） | 无物性 | 否 |
| `exp_segvggt_finetune_agnostic_full/2026-07-29_20-35-30`（20G） | 无物性 | 否 |

（`exp_segvggt_physgm` 另三个时间戳目录里 **0 个 ckpt**，无需标记。）

已做：目录改名为 `2026-07-29_07-35-57__STALE-PHYSGM-NORM`，内放
`STALE_NORMALIZATION.md` 写明旧常数三行、新常数三行、以及"跑 `segvggt_infer.py`
会静默得到错好几个数量级的 E"。全仓无任何文件引用该路径；`output/` 在 `.gitignore:180`
⇒ 改名不动 git。

**没做**（Q4 的 (b)）：把常数哈希写进 ckpt、推理时不匹配硬报错。用户选了轻档。
代价诚实记在这里：**这道护栏是目录名，只挡人不挡程序**——谁把 ckpt 拷到别处，标记就没了。
"抄来的常数"这个类型的错误因此仍能复发，只是这一份具体的 ckpt 被挡住了。

## 完成判据核对

- 常数来源定死并写进代码：✅ `parsers.py:74`，注释写明拟合来源 / 日期 / 脚本命令 /
  "换数据集要重新拟合，不能照抄这三行"。
- `physgm_denormalize` 与训练用同一套：✅ 同一个 tuple，一处定义。
- 按训练集划分重生成：✅ `physgm_norm_infinigen_train.json`（**注意文件名带 `_train`**——
  data 盘根下那份旧的 `physgm_norm_infinigen.json` 是全量 1466 场景拟合的，不是在用的那份）。

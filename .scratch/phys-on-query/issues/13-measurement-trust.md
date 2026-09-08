# 13 — 度量装置不可信（val 抽样、clut 口径、控制台行、权重保留）

Type: task
Status: closed (2026-09-08)
Blocked by: —
Assignee: claude (opus_phys_map session, 2026-09-08)

> 由 [01 号票](01-unfreeze-lora.md)的实测过程逼出。四个缺陷独立，主题相同：
> **这套 val 度量能读出的，比它看起来能读出的少得多。**
> ⚠️ **13.1 和 13.4 必须在下一次起跑之前做完。**

## 前提：跨 step 的形态四次骗了人

| 出现在 | 形态 | 怎么被推翻 |
|---|---|---|
| s3500 | A 的 E 三点连降到 0.944 | 下一点回弹；B 同点也是 0.946 ⇒ 共动 |
| s6500 | A 三项首次同时 <1（0.880/0.966/0.974） | 一步回弹；B 同点也下探 |
| s8000–9000 | A 连三点三项 ≤1、E 守在 0.95 下 | s9500 回到 0.974；**B 在 s8000 是 0.985/1.024/1.066，没跟** |
| s4000–5000 | "A 的分割见顶了"（ap50 连跌两点） | s5500 三项创新高 |

⇒ **少于十点的窗口不可信。** 站得住的只有**同 step 跨臂**和**分段（≥10 点）均值**。

## 13.1 val 视角没固定 seed ⇒ 跨 step 趋势不可读

`fixed_views_and_shape: true` 在 `src/dataset/data_sampler.py:318-321` 只把**视角数**和
**分辨率**压成退化区间，**选哪几帧仍随机**。⇒ `phys_n_matched` 在 27.11–30.17 浮动，
`*_const` / `*_clut` 每点都不同。

噪声量级有一把免费的尺：A 臂底座与 geo 头全冻，而 geo 头只吃 aggregator token
（`src/model/segvggt/models/segvggt.py:124`，不经 instance 分支）⇒ **A 的 geo 输出是常数函数，
`val/geo_*` 的跨 step 变化全部来自重采样**。实测 `geo_camera` 0.2012 / 0.2065 / 0.2298（14%）。

⚠️ **它修不好什么**：聚合后 `*_const` 只摆 ±2.2%（40 点 0.7448–0.7785），而学生/clut 摆 ±6%。
钉 seed 消掉的是前者。真正价值是让不同 checkpoint 站同一把尺。

**做**：给 val 视角选择一个与 step 无关的固定 seed，train 侧不动。

## 13.2 clut 基线两套口径，差三倍

| 口径 | clut vs const | 出处 |
|---|---|---|
| val（场景加权，只覆盖匹配上的 ~29 实例/场景） | **E −9.7% / ρ −4.3% / ν −5.8%** | 本轮 40 点 |
| 留出集 pooled（2724 实例） | −17.4% | 05 号票 |
| 全语料 pooled（51,981 条） | −15.6% | 04 号票 |

不是同一个统计量（实例加权 vs 场景加权）。**主表报哪个必须先定死**，否则"学生比查表好 3.3%"
的分母会随口径变。

**做**：选一个；另两票的数按同口径重算或标注不可比。

## 13.3 控制台那行是单个 batch

`src/model/wrapper/segvggt_wrapper.py:393` 的 `batch_idx % 100 == 0` 使它只在 batch 0 打印，
而 val 每 rank 只有约 10 个 batch ⇒ **永远只报第 0 个 batch（2 场景、~36 实例）**。
A 的最后一个 val 点：控制台 E/clut = **1.447**，聚合值 **0.952**。本轮曾据此写下
"学生劣于常数基线"的错误结论。

**做**：改报聚合值（或不打）；监视器改读 tfevents。

## 13.4 中间 checkpoint 被删光

`save_top_k: 1` + `every_n_train_steps: 1000`（`phys_query_arm_a_frozen.yaml:97-99`、
`phys_query_arm_b_lora.yaml:108-110`）⇒ 每存必顶掉旧的，两臂最终各只剩 `step_20000` + `last`。
代价：**用固定视角离线重评所有中间 ckpt、一次性拿到干净曲线**这条路对本轮永久关闭。

**做**：起跑前定策略。每份 10.5 G，全留 20×2=420 G 装不下（现余 709 G）；
可每 2000 步留一份（231 G），或 `save_weights_only: true` 后每份约 5.5 G。

## 13.5 附带：监视器造过一条不存在的读数

收到过 `[A] step=18000 n=38.0 const 0.8625 clut 0.9219 E 1.0244`，**两臂日志里都没有**
（A 的 s18000 唯一一行是 n=41.0 / const 0.8091 / E 1.1624）。其基线恰等于 B 的 s12000 抽样，
学生值对不上任何一行。影响有限（所有数字都是重读 tfevents 得到的），但它会凭空造数。

## 完成判据

1. 两次 validation 的 `phys_n_matched` / `*_const` 完全相同；
2. clut 口径定死一个，04/05 标注同口径或不可比；
3. 控制台行报聚合值，监视器读 tfevents；
4. 下轮起跑前保留策略写进 config 并算过盘容量。

## 明确不在本票内

- **改指标定义本身**（换实例加权、换匹配规则）——会让本轮 40 点与未来不可比，单独判。
- **重跑本轮补权重**——7–9 h/臂，而 [01](01-unfreeze-lora.md) 的结论建立在不受影响的
  同 step 跨臂比较上，不值得。

---

## 答复（2026-09-08，本票关闭）

**四条全部落地，并且在验证过程中当场撞出第五条——比原来四条里的任何一条都严重。**

### 13.1 val 视角固定 ✅

不是"给 val 一个固定 seed"，是**给每个场景一个由它自己的名字决定的 seed**：
`ViewSampler.scene_generator()`（`src/dataset/view_sampler/view_sampler.py`）在
`stage != "train"` 时返回 `torch.Generator`，种子 = `sha256(scene_id)` 前 8 字节
（不用 `hash()`：它每进程加盐，重启就换一套视角）；`ViewSamplerBoundedFixed.sample()`
的 5 个随机点全部改成吃这个 generator，train 侧返回 `None` ⇒ 原样走全局 RNG，抖动保留。

比"一个固定 seed"强的地方：视角不再依赖 step、rank、worker 数、batch 顺序，
**离线重评一份 ckpt 拿到的也是同一批帧**——13.4 留下来的中间权重才有意义。

实测（4 视角 / 252×448 / 2 卡 / limit_val_batches=4）：连续两次 validation 的
`phys_n_matched` 36.1、三项 `*_const` 0.9249 / 0.2772 / 0.0498、三项 `*_clut`、
以及 6 条 `val/geo_*` **逐位相同**；学生值只在第 4 位小数上动（那两步真的训了）。
⇒ 完成判据 1 成立。固化成 `tests/test_val_view_determinism.py`（5 个测试，毫秒级）。

### 13.2 clut 口径 ✅ 定为**实例加权 pooled**

理由不是"val 那个不好看"，是：04/05 报的都是 pooled，而**这张图上所有刻度都是它们给的**；
且场景加权的均值取决于实例怎么散在场景里，换个划分就变。

但**没有改动任何现有 tag**（票面明确把"换实例加权"划出本票，改了本轮 40 点就废）：
`aggregate_rows()` 额外吐出 `<tag>_xn = 每场景 MAE × 该场景实例数`，于是

    pooled MAE = mean(<tag>_xn) / mean(phys_n_matched)

只用均值就能还原 pooled——Lightning 只会做均值，这一条是关键。
原有 `phys_mae_*` 保持场景加权，继续当在线曲线。两个口径实测确实分岔
（学生 log10 E：场景加权 0.9227 vs pooled 0.9723）。
控制台多打一行 `[val phys pooled ...]`。固化成 `tests/test_physics_metric_calibers.py`。

⇒ 04（全语料 pooled −15.6%）、05（留出集 pooled −17.4%）与主表**同口径，可比**；
本轮 40 点里的 `−9.7%` 是场景加权，**标注为不可比**，不进主表。

### 13.3 控制台行 ✅

三条 `logger.info`（instance / phys / geo）全部从 `validation_step` 挪到新的
`on_validation_epoch_end`，直接读 `trainer.callback_metrics`——**打印的就是写进
tfevents 的那个数**，batch 0 那条路彻底没了。`batch_idx % 100 == 0` 的图像记录保留
（它本来就该是单个 batch）。

### 13.4 中间 checkpoint ✅ 两条序列

一条序列同时要"能续跑"和"是曲线"，必然互相顶。拆成两条（`src/main.py` 第二个
`ModelCheckpoint`，新配置项 `checkpointing.snapshot_every_n_train_steps`）：

| 目录 | 内容 | 实测大小 | 数量 |
|---|---|---|---|
| `checkpoints/` | 全量（含 AdamW 状态），`save_top_k: 0` ⇒ 只剩 `last.ckpt` | **10.52 G** | 1/臂 |
| `snapshots/` | `save_weights_only: true`，全留 | **6.62 G** | 10/臂（每 2000 步） |

`save_top_k` 从 1 改成 0：原来那份 "best by step" 和 `last.ckpt` 逐字节是同一个东西。
**盘账（实测非估算）**：132 G + 21 G ≈ **153 G**，`/mnt/storage_pool` 现余 **531 G**
（不是票面写的 709 G——第一轮自己的产物还在盘上）。⇒ 完成判据 4 成立。

顺带核对了一件事：全量 ckpt 两臂差 **0.076 G**，恰好等于 9.44M LoRA 参数 × 2 个 fp32 动量
⇒ 优化器状态确实是 8 B × 可训参数，01 号票数的参数量再一次对上。

### 13.5 监视器 ✅（规则，不是代码）

**唯一可引用的读数来源是 `scripts/read_tfevents.py` 读出来的 tfevents**；控制台行只是方便，
不是来源；记不清就重读，不许复述。这条已写进 `read_tfevents.py` 的模块 docstring。
13.3 落地后控制台与 tfevents 同源，两者不再可能互相打架。

---

## ⚠️ 13.6（本票新查出，比上面四条都重）：本轮 40 点只覆盖了 74 个验证场景里的 **18 个**

`self.log(...)` 默认 `sync_dist=False`，Lightning 只从 rank 0 写日志
⇒ **写进 tfevents 的是 rank 0 自己那份局部均值**。铁证：`val/phys_n_matched`
在 s499 是 `28.944444…` = **521 / 18**，而 74 场景 / 4 卡 `drop_last` 后每卡正是 18 个。

已给 val 侧全部 5 处 `self.log` 加上 `sync_dist=True` ⇒ 覆盖面从 18 变成 **72/74**，
噪声按 √4 缩。这不是换指标定义（同一个估计量，只是样本大了），本轮 40 点仍可比，
只是**必须记作"18 场景的读数"**。

**这条不推翻 01 号票**：两臂的 rank 0 吃的是同一批场景（`DistributedSampler(shuffle=False)`
+ 13.1 之后连帧都一样），01 的结论建立在同 step 跨臂比较上，那个比较不受影响。
受影响的是**绝对数字的置信度**：票面前提里"少于十点的窗口不可信"，现在知道了其中一半原因。

### 13.6 的连带：一个 `sync_dist` 一开就炸的潜伏 bug（已修）

加上 `sync_dist=True` 后 2 卡实测立刻读出**离谱但看着像真的**的数：
`nu` 的常数基线 0.5285（真值 0.0498）、`log10 E` 学生 0.4415（真值 0.9227），
而 1 卡跑同样的代码全部正常。

原因：`compute_physics_metrics_batch` 里 `keys = {k for r in rows for k in r}` 是个 **set**，
Python 字符串哈希**每进程加盐** ⇒ 两个 rank 以不同顺序 `self.log` 同一批指标；
Lightning 按**插入顺序**做 all-reduce ⇒ 集合通信配对错位，**每个 rank 拿回别人的数**。
修法是 `sorted()`，三处 val 日志循环全部排序，详细注释放在 `_log_geo_drift` 上。
排序后 2 卡与 1 卡逐位一致。

⚠️ **这个坑是通用的**：任何 `sync_dist=True` 的日志循环，只要迭代顺序在 rank 间不确定，
就会静默地串号。它不报错，数字也在合理范围里——本票是靠"未训练的头应该等于常数基线"
这条免费的自检抓到的。

## 明确没做

- **不重跑本轮补权重**（票面已判）。中间权重对本轮永久没了，13.4 只对下一轮生效。
- **不改指标定义**：匹配规则、加权方式、`phys_mae_*` 的含义一律没动。
- 迁移期只改了 `bounded_fixed` 一个采样器（本图唯一在用的）；
  `scene_generator()` 放在基类上，别的采样器要用自己接。

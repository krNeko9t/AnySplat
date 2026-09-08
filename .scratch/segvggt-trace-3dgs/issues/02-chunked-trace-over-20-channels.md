# 02 — 让 trace 支持 feat_dim > 20（分趟，不重编译）

Type: task
Status: resolved
Blocked by: —
Blocks: [03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)
Assignee: krNeko9t

## Question

`scripts/trace_instance_to_gaussians.py:1495` 现在是**硬拒**：

```python
if feat_dim > TRACE_CHANNELS:
    raise ValueError(f"feat_dim={feat_dim} > TRACE_CHANNELS={TRACE_CHANNELS}. ...")
```

而 SegVGGT 的 feature map 是 **128 维**（地图 F1）。要么重编译 CUDA，要么分趟。
**已判分趟**（见地图 Out of scope）：通道之间在 trace 里完全独立，
`gau_sem[:, c]` 只依赖 `img_sem[:,:,c]`，所以把 128 切成 `ceil(128/20) = 7` 段各跑一趟、
按通道拼回去，**结果与一次 128 通道的 trace 逐字相等**。

## 要改的

- `src/trace_render/trace_rasterize.py`：`trace_single_view` 目前在 `:217` 建
  `img_sem = torch.zeros(h, w, TRACE_CHANNELS)`，把 feat 塞进前 `feat_dim` 个通道。
  加一层按 20 分段的循环，返回拼好的 `[P, feat_dim]`。
- `scripts/trace_instance_to_gaussians.py:1495`：把硬拒换成分趟。
- `num_gsem` 每趟都一样（几何相同、`img_mask` 相同）⇒ **只累加一次**，别累加 7 次，
  否则归一化的分母大 7 倍。这是本票最容易埋的坑。

## 判据

- 拿一个现成的低维场景（`bench_gt_idmap.json`，`id_embed_dim=16`）跑新旧两条路，
  `gau_sem` 应当**逐元素相等**（同一份几何、同一份 mask，浮点累加顺序也没变）。
- 造一个 128 维输入，确认分 7 趟跑得完、`num_gsem` 与 20 维那次一致。
- 记下 7 趟相对 1 趟的**墙钟开销**。若慢到不可接受，地图 Out of scope 里
  「重编译 TRACE_CHANNELS」那条允许带着这个数字重开。

## 备注

有个精确的省钱法，本票**不做**，只记下来：若在 trace 之前就定死要问的 query 集 `Q_all`，
则只需 trace `r = rank(span(Q_all))` 个通道就能**无损**保留所有 `q·f`。
M ≤ 20 时一趟搞定且严格等价。它要求 03/04 的顺序反过来，现在不值得为它重排。

## Answer

**分趟做完了，等价，7 趟的代价就是 7×，绝对值可忽略。放行 03 号票。**

### 改了什么

- `src/trace_render/trace_rasterize.py` 新增 `trace_single_view_chunked(...)`：
  收 `feat_hwc [H,W,D]`（D 任意），切成 `ceil(D/20)` 段各跑一趟 `trace_single_view`
  （最后一段补零），沿通道轴 `cat` 回 `[P, D]`。
  `num_ray` / `radii` / `out_color` **只取第一趟的**，不累加——本票埋的坑在这里躲掉了。
- `scripts/trace_instance_to_gaussians.py`：删掉 `:1495` 的硬拒和调用点的手工 padding，
  改调 `trace_single_view_chunked`；打印行现在报 `TRACE_CHANNELS=20 x N pass(es)`。
- 新增 `scripts/check_chunked_trace.py`（判据脚本，可复现；场景走 `$CHUNKED_TRACE_SCENE`，
  默认 `.../instascene_preprocessed_data/3dovs/bench`）。

### ⚠️ 判据本身要改：trace kernel 不是逐位可复现的

本票原来写「`gau_sem` 应当**逐元素相等**……浮点累加顺序也没变」。**这一句是错的。**
kernel 用 `atomicAdd` 累加，**同一条路、同一份输入跑两遍就已经不相等**：

| 对照 | maxabs | mean |
|---|---|---|
| **控制组**（同一条路两遍） | 3.4e-3 | 6.8e-8 |
| D=16 新 vs 旧 | 3.8e-3 | 6.6e-8 |
| D=20 新 vs 旧 | 3.3e-3 | 6.5e-8 |
| D=128 分 7 趟 vs 手工 7 段拼接 | 5.5e-3 | 6.7e-8 |

`gau_sem` 的量级 absmax = 2.0e+3 ⇒ **底噪/量级 = 1.7e-6**。
新旧之差与控制组同一量级 ⇒ **等价成立到 kernel 自身的非确定性为止**，这是能拿到的最强结论。
`num_ray` 和 `out_color` 则是**逐位相等**的（整数计数 / 与 `img_sem` 无关）。

**带给下游**：整条 trace 链路都有 ~1.7e-6 的相对底噪，
[04 号票](04-query-pooling-and-3d-dedup.md)定去重阈值时别定到这个尺度以下
（远比 01 号票那个 3% 的 norm 地板细，实际不构成约束，记下即可）。

### 判据逐条

1. **低维新旧一致**：D=16 / D=20（`bench_gt_idmap.json` 用的是 `id_embed_dim: 20`，
   不是本票原文写的 16，两个都测了）**PASS**，`num_ray`/`out_color` 逐位相等。
2. **128 维跑得完**：`[1045236, 128]`，7 趟，与手工逐段拼接一致 **PASS**。
   `num_ray` 与 D=20 那趟**逐位相等**（sum=35,809,772；若误累加 7 次会是 250,668,404）。
3. **墙钟**：1 趟(D=20) 23.5 ms/view，7 趟(D=128) 168.5 ms/view，**比值 7.07–7.16× ——
   就是线性，没有额外常数开销**。外推 bench 36 视角 = 6.1 s，garden 185 视角 = 31.2 s。
   ⇒ **完全可接受**，地图 Out of scope 里「重编译 `TRACE_CHANNELS`」那条**不重开**。

### 端到端（真脚本，不是 harness）

`scripts/trace_instance_to_gaussians.py --feature_source gt_idmap --max_views 8` 在 bench 上
跑了 `--id_embed_dim 20` 和 `128` 两遍（128 在改之前是硬报错）：

- 两遍都 `Gaussians traced: 764,178 (73.1%)`，`num_ray mean=359.1 max=466386` **完全一致**
  ⇒ 分母没被乘 7 的又一个独立旁证。
- D=128 的 `gt_idmap` 解码照样recover 出同样的 7 个 ID ⇒ 语义信息穿过 7 趟没丢。

`tests/test_trace_render_cluster_paths.py` + `tests/test_trace_cameras_transforms.py` 8 项全过。

### 本票未做的

- 没给 `tests/` 加用例：判据要 CUDA + 一个真实高斯场景，做不成不依赖环境的单测；
  可复现的入口是 `scripts/check_chunked_trace.py`。
- 备注里那个「只 trace `rank(span(Q_all))` 个通道」的省钱法仍然不做——
  7 趟 = 31 s，已经没有省的价值了。

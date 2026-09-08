# 02 — 让 trace 支持 feat_dim > 20（分趟，不重编译）

Type: task
Status: open
Blocked by: —
Blocks: [03 — 把 SegVGGT 接成 trace 的 feature source](03-segvggt-feature-source.md)
Assignee: —

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

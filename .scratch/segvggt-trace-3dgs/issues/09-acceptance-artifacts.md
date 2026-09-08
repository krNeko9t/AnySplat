# 09 — 验收交付什么：文件包 vs 现场可视化

Type: grilling
Status: open
Blocked by: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Blocks: —
Assignee: —

> 这是你开图时自己提的那个问题（"是现场可视化？还是给一堆 output 文件？"），
> 挂到 04 之后 —— 先看东西长什么样，再定怎么交。

## Question

地图 Destination 定的是"一张能进论文的定性图 + 一份可复现脚本"。
本票把它落成具体的文件清单。候选：

1. **文件包**：
   - `instances.json`：实例 id / 高斯数 / score / (E, ν, ρ) SI 值 / 来自哪些 query
   - `instance_ids.npy`：`[P]` 每个高斯的实例归属（-1 = 无人认领）
   - `per_instance/*.ply`：按实例切开的高斯（`postprocess_seg3d_split` 已有类似能力）
   - `renders/*.png`：几个视角的上色渲染
2. **现场可视化**：一个能转能点的 viewer（点一个实例弹出它的物性）。
   演示效果最好，但 17 天里是纯消耗，且论文里放不进去。
3. **两者都要**，viewer 只做最薄的一层（复用现成的高斯 viewer，喂 `instance_ids.npy` 上色）。

## 倾向（待你定）

**1 为主，3 的 viewer 部分只在 04 顺利收工后有余力才做。** 论文要的是 png，
不是 viewer；而文件包同时喂饱了 07 号票的指标计算和图的生成。

## 还要定的

- 输出目录结构跟 `trace_output/<scene>_<tag>/` 的现有约定走，还是另起？
- 落盘 `gau_feat [P,128]` 吗？（P≈10⁶ ⇒ fp16 也有 256 MB/场景。
  留着能反复换 query 不重跑 trace，不留则每次改阈值都要重跑 7 趟。）

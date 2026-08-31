---
id: R2
title: 跨视角 instance id 是否真的对齐
type: wayfinder:research
status: open
assignee: -
blocked-by: []
---

## Question

$L_{cross}$（Eq.7）建立在一个硬前提上：论文原话是 "GT instance mask **with instance IDs
aligned across views**"。如果同一物体在不同帧里 id 不同，这一项不但没用，还会**主动把不同
物体的原型拉到一起**——比 3.2 节整个不做还糟。

在本仓库内查清楚（不要问人，脚本和 manifest schema 都在仓库里）：

1. `scripts/make_manifest_inscene_scannetpp_v2.py` 与 `scripts/make_manifest_inscene_re10k.py`
   分别是怎么产生 `instance_mask` 的？id 是**场景级全局**的，还是**逐帧独立**的？
2. `re10k` 子集尤其可疑——它是 IGGT 用 SAM + tracking 造的伪标注，跨帧一致性可能明显弱于
   `scannetpp_v2`（后者来自 ScanNet++ 的 3D 实例标注投影，天然全局一致）。查证这个猜测。
3. `src/dataset/` 里读 manifest 的那条路径，有没有对 id 做过任何重映射/压缩？
   （重映射如果是**逐帧**做的，会把原本全局一致的 id 打散。）
4. `instance_valid_mask` 在这两个子集上分别覆盖多少像素？id 0 是「真背景」还是「未标注」？
   （`scripts/check_instance_id0.py` 已有，读它的判据即可，不必真跑。）

**产出**：两个子集各自的结论（全局一致 / 逐帧独立 / 部分一致）。如果 `re10k` 不一致，
给出建议：是 (a) 只在 `scannetpp_v2` 上开 $L_{cross}$、(b) 整个 re10k 不用、
还是 (c) 用某种 per-sample 开关。这个结论会直接改写 T2 和地图的数据配比。

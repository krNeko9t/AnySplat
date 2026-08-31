---
id: T11
title: 跨帧 instance id 一致性实测（re10k + scannetpp_v2）
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

R2 判定：`scannetpp_v2` 的跨帧 id 全局一致是「设计假定 + 全链路自洽」，`re10k` 是
「未验证的假定一致」（SAM + tracking 伪标注，track 断裂 / id 切换风险）。本票把这个假定
变成实测数字。

写一个抽查脚本（可放 `scripts/`，CPU 可写可自测），对两个子集各采样若干场景：

1. 同场景任取两帧，比较 id 集合的重叠率（|id 交集| / |id 并集|，排除 id 0）。
2. 弱一致性指标（无需精确配准）：逐 id 的 mask 面积跨帧稳定性、id 直方图的相关性；
   有条件时可用已知的相机位姿把一帧 mask 投影到另一帧比对（manifest 里有相机参数）。
3. 顺带跑 `scripts/check_instance_id0.py` 的判据，确认两子集 id 0 是「真背景」。

**产出**：每子集一个一致率数字 + 一句裁决。若 `re10k` 一致率 > ~95%，T2 可对 `re10k`
打开 $L_{cross}$（撤掉 R2 建议的 per-subset 关闭）；否则维持关闭。

**HITL**：脚本本机可写，但运行需要 `/mnt/shared-storage-gpfs2/...` 数据（本机未挂载），
要人在有数据的机器上执行。

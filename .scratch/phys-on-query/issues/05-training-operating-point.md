# 05 — 三周内能跑完的运行点

Type: task
Status: open
Blocked by: 01, 02

## Question

距 2026-09-25 只剩三周，且这个 checkpoint 必须留出重跑一两次的余地。要定：

1. 硬件：几张什么卡、能连续占多久。
2. 数据量：用 Infinigen 全量还是子集；每个场景多少视角。
3. query 数 Q、batch、步数——要能估出一次完整训练的墙钟时间。
4. **课程**：GroupForward 的教训是"先几何 → 再实例 → 再联合"，去掉 curriculum 是它全表最大的
   单项退化（mIoU 0.5989→0.4669，PSNR 26.23→18.92）。我们的起点是 pretrained SegVGGT，
   等于前两阶段已经完成 ⇒ 大概率只需要"联合"这一段，但要确认。
5. **几何保险丝**：`segvggt_geo_supervision: teacher`（frozen VGGT 蒸 depth/camera）要不要开。
   当前 joint 配置用的是 `gt` 且 `segvggt_geo.weight: 0.0` = 没有任何几何约束，
   而 01 修完之后梯度真的会进主干 —— 这个保险丝的必要性随之上升。

**完成判据**：一份能直接启动的配置 + 一个墙钟时间估计。

# 05 — 三周内能跑完的运行点

Type: task
Status: open
Blocked by: —  (01、02 均已关闭；(b) 臂的 config 另需 09)

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

## 2026-09-07 更新（[01 号票](01-unfreeze-lora.md)关闭后）

**阻塞解除。但本票的形状变了：要给两臂定运行点，不是一臂。**
01 号票判的是 **(d) 两臂并行对照**——4 卡跑全冻底座、4 卡跑放开 LoRA，同时起，
唯一变量是一条 `!*.lora.*`（+9.44M / +1.9%）。

01 号票已经交掉了本票原来最难的两项，别重新推导：

- **第 1 项（硬件）与第 3 项（墙钟）已有实测底**：`bms-39468022-001`，8×A100-40G，
  2026-09-07 核实 0 MiB 占用，env `anysplat`。2026-07-28 那次 joint 跑实测
  **1.30 s/step**（8 卡 / bs=1 / 4 视角 / 252×448）⇒ 20k step ≈ **7.2 小时**。
  两臂各占 4 卡 ⇒ 每臂约 **14.4 小时**，一天内两条曲线同时到手。
  距 09-25 还有 19 天，**算力不是约束**——本票不必再为省算力做取舍。
- **第 5 项（几何保险丝）已判**：**不开 teacher，`segvggt_geo.weight` 保持 0**。
  01 号票 Q2 判定几何不作为本图交付（终点只写 class + P），代之以
  [11 号票](11-geo-drift-readonly-metric.md) 的**只读** depth/pose 漂移指标。
  本票不要把这条推回去。

**本票仍要定的**：第 2 项（数据——scannet100 100 条 vs Infinigen 全量 1466 条，
每场景多少视角）、第 3 项的 Q/batch/步数、第 4 项（课程）。
**两臂必须共享这些设置**，否则 01 号票整个对照作废。

⚠️ **开跑前的硬门槛**：任何 `freeze_keywords` 改动都要重生成
`config/experiment/locks/<X>.lock`，缺 lock 或 diff 不符 = 启动硬错
（`freeze-contract` 图已落地）。(b) 臂那份 lock 的 diff 应**恰好 192 行**从 `-` 翻成 `T`。

**完成判据**：**两份**能直接启动的配置（(a) 臂可今天就绪，(b) 臂等 [09 号票](09-freeze-matching-language.md)）
+ 各自的 lock + 一个墙钟时间估计。

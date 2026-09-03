# 02 — 把 Infinigen 全量 VLM 伪标签接进训练

Type: task
Status: open
Blocked by: —

## Question

标签已经生成（Infinigen 完整数据集，naive 视角选择 + `pred_phys.prompt`），
但**不在本机**。要开训必须先把这条打通。

需要落定的事实（做完记在 `## Answer` 里，后面的票都依赖它们）：

1. 标签在哪（机器 / 路径 / 大小），怎么进到训练机。
2. 每条记录的实际 schema：至少要有 `(scene, instance_id, E, ν, ρ)`，以及能拿到物体类别的字段
   （Infinigen 的 Factory 类名）。类别字段是 04 的前提。
3. 单位与归一化：三个量各自的单位是什么，`log10 z-score` 的 mean/std **从哪个集合上算**
   （必须是训练集，且要存下来，否则推理期反归一化对不上）。
4. 覆盖率：多少实例有标签、多少缺失、缺失怎么处理（跳过 / 置 unknown / 不参与 loss）。
5. 与实例 GT 的对齐：伪标签的 `instance_id` 和分割 GT 的 id 是不是同一套。

**完成判据**：能在训练机上加载一批数据，打印出若干个实例的 (类名, E, ν, ρ) 且数值合理。

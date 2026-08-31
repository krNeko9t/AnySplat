# InstanceSplat 复现地图

> `wayfinder:map` — 本地 markdown tracker。票在 `tickets/`，一票一文件。
>
> **认领**：把票头的 `assignee: -` 改成你的标识，**动手之前先改**，并发 session 才会跳过它。
> **关闭**：`status: closed`，并在票尾追加 `## 解决` 段（答案写在票里，不写进本文件），
> 然后回本文件的「Decisions so far」加一行指针。
>
> **查前沿**（open + 未认领 + blocked-by 全已关闭）：
> ```bash
> .scratch/instancesplat/frontier.sh
> ```

## Destination

本仓库里有一套**能在 8×A100 集群上跑起来**的 InstanceSplat 路线：从 `hf:lhjiang/anysplat`
权重起步，实例特征经**可微渲染回 2D** 后用原型式对比 loss 监督（$L_{pull}/L_{push}/L_{cross}$），
外加边界感知 RGB loss（$L_{bd\text{-}rgb}$），backbone 全量微调，超参/分辨率/视角数按论文对齐。
终点 = 跑完一次全量训练，并且有**免聚类的三个诊断标量 + HDBSCAN 里程碑指标**能判定
「训练到底修好了没有」。决策与实现**都在地图内**。

## Notes

- **执行也在地图内**：本 effort 覆盖 wayfinder 的「plan, don't do」默认。票可以是实现票，
  不只是决策票。但每 session 仍然只解一张票（research 票除外）。
- **领域**：前馈 3DGS + 实例分割。论文原文在 `ref_knowledge/InstanceSplat/InstanceSplat.md`
  （公式编号 Eq.1–17 直接引用它）。仓库现状看 `docs/repo_knowledge.md` 第 2 节与「算法登记表」。
- **每 session 应该 call 的 skill**：`grilling` + `domain-modeling`（默认）；
  写代码前看 `docs/layered_scheme.md` 的四层契约。
- **硬约束：不干涉既有实现**。`instseg_anysplat.yaml`、`loss_disc.py`、`loss_mvc.py`、
  `config/model/encoder/anysplat.yaml` 的既有字段语义、SegVGGT 整条线——**一律不动**。
  新能力 = 新 loss 文件 + 新 experiment yaml + `EncoderOutput` / `DecoderOutput` 上默认
  `None` 的新可选字段。
- **本机无 GPU、无数据集**（`/mnt/shared-storage-gpfs2/...` 未挂载）。所有代码票的验证
  = CPU 合成张量；所有跑数票（T8/T9/T10）是 HITL，要人在集群上执行。
- **不要引入论文没有的超参**。凡是偏离论文的取值，必须在票的 `## 解决` 里写明**为什么偏离**。

### 已锁定的配方（charting session 的共识，不再重开）

| 项 | 值 | 来源 |
|---|---|---|
| 载体 arch | `anysplat`，起始权重 `hf:lhjiang/anysplat` | 论文 3.1 结构同构 |
| 实例特征维度 | 8 | 论文 3.1 |
| backbone | **解冻**，transformer lr × 0.1 | 论文 4.1；Table 3「3DGS frozen」消融 |
| 算力 scaling | 8×A100 40G，有效 batch 4×、lr `2e-4`、iters `10k`、warmup `1k`、cosine | sqrt scaling，偏离已知且可归因 |
| 监督视角 | `num_target_views: 0`，2–8 个 context view 全部既输入又监督 | 论文 Eq.11 求和上标为输入视角数 |
| 分辨率 | 长边 448，宽高比五档离散 {0.5, 0.625, 0.75, 0.875, 1.0} | 论文 4.1 是连续采样；离散化是为 40G 显存可预测性，属**有意偏离** |
| 数据 | InsScene-15K 的 `processed_scannetpp_v2` + `processed_re10k` | 论文 4.1 |
| loss 权重 | λ_ins=0.01, λ_bd=0.02, λ_p=0.05, (λ_pull,λ_push,λ_cross)=(2,1,2) | 论文 4.1 |
| $b_i$ 梯度 | **detach** | 论文 3.3「detached in their respective coupling paths」；不 detach 存在抹平 $S_i$ 的作弊通道 |

## Decisions so far

<!-- 一行一张已关票：- [票名](tickets/xxx.md)：答案一句话 -->

- [实例 margin 的取值](tickets/R1-margin-取值.md)：δ_pull=0.1 / δ_push=1.0 / δ_cross=0.05（球面几何 + IGGT 同组 M=1.0 先例 + 本仓 mvc 先例的有据外推，非论文原值）
- [跨视角 instance id 是否真的对齐](tickets/R2-跨视角id是否对齐.md)：scannetpp_v2 全局一致、re10k 部分一致（未验证）→ L_cross 按数据源 gating，re10k 只参与 pull/push；实测脚本见 T11
- [40G 显存装不装得下](tickets/R3-40G显存预算.md)：每卡 1 场景 × 8 视角 @448 可行（峰值 ~28–32GB），必改 `max_img_per_gpu: 24→8`；checkpointing 已硬编码，OOM 砍序 = 视角→6 → 分辨率→384 → voxel_size→0.004

## Not yet specified

- **训练不收敛时的应对**：等 T9 smoke run 的曲线出来才知道要不要动 warmup / margin / λ_ins。
- **显存不够时的退路**：砍序已定（视角→6、分辨率→384、voxel_size→0.004，见 R3）；是否真需要，
  等 T9 smoke run 的实际占用。
- **`voxel_size` 要不要调**：现在是 AnySplat 默认 `0.002`，论文没给这个数。体素粒度直接决定
  实例特征的空间分辨率（太粗会糊掉小物体的实例边界），但要有渲染结果才能判断。
- **8 视角下高斯数量与渲染显存的关系**：`voxelize` 后的高斯数随视角数增长，两路渲染
  （RGB + 8 维特征）的峰值显存曲线未知。

## Out of scope

<!-- 一行一条：越过目的地的工作，附一句为什么，链接已关的票（如果它曾是票） -->

- **语义分支全套**（LSeg teacher、$L_{sem}$、语义引导硬负样本 η、推理侧实例级语义聚合）：
  训练侧其实可做（teacher 特征从 RGB 前向得到，不需要语义标签），但手上没有语义标注 →
  **没有 mIoU 可验收**，做了判不了对错；且论文 Table 1 显示 η 的增益（65.01→65.23）在噪声量级。
- **论文 Table 1 口径的评测**（ScanNet v2 的 50 个 held-out 场景、开放词汇 mIoU/mAcc）：
  缺 ScanNet v2 数据与语义标签。
- **`processed_infinigen` 子集**：论文没用（合成域），引入会让「效果变差」无法归因。
- **SegVGGT 路线**（算法 #6）：集合预测范式，原型对比学习在它上面无处安放。
- **改动既有实例路线**：`instseg_anysplat.yaml` / `loss_disc.py` / `loss_mvc.py` 保持原样。

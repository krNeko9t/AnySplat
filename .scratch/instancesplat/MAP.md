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
| 数据 | InsScene-15K 的 `processed_scannetpp_v2` + `processed_re10k`（**re10k 当前加载不了，见 T13**） | 论文 4.1 |
| loss 权重 | λ_ins=0.01, λ_bd=0.02, λ_p=0.05, (λ_pull,λ_push,λ_cross)=(2,1,2) | 论文 4.1 |
| $b_i$ 梯度 | **detach** | 论文 3.3「detached in their respective coupling paths」；不 detach 存在抹平 $S_i$ 的作弊通道 |

## Decisions so far

<!-- 一行一张已关票：- [票名](tickets/xxx.md)：答案一句话 -->

- [实例 margin 的取值](tickets/R1-margin-取值.md)：δ_pull=0.2 / δ_push=1.0 / δ_cross=0.3
  （ℓ2 归一化后的欧氏弦长）。δ_push=1.0 **不是新超参**——`config/loss/mvc.yaml` 已在用，
  且 IGGT 在同样归一化的 8 维实例特征上用 M=1.0、λ 也是 (2,1)。δ_pull / δ_cross 无任何
  论文或代码来源，是约束区间内的取点（建议区间 [0.1,0.25] 与 [0.2,0.4]），**待 T9 实测校准**。
- [跨视角 instance id 是否真的对齐](tickets/R2-跨视角id是否对齐.md)：「re10k 逐帧独立」的猜测
  **被证伪**（是 SAM2 track id，构造上全局），`src/dataset/` 全链路零重映射，票里担心的
  「最隐蔽的坑」不存在。但 re10k 的 **id 碰撞残余率无人量化** → $L_{cross}$ 只在
  `scannetpp_v2` 上开，用 `spp_`/`re10k_` 场景名前缀在 loss 层内部逐样本门控。
- [修复 aggregator bf16 强制转换](tickets/T11-修复aggregator-bf16强制转换.md)：**已修**。
  `aggregator.to(torch.bfloat16)` 改由新增可选字段 `aggregator_param_dtype`
  （`Literal["bfloat16","float32"]`，**默认 `bfloat16` = 现状**）驱动，两个既有 yaml 一字未改。
  实测确认 bug 真实存在，且是「按权重量级**选择性**冻结」而非全冻结：`lr=2e-5` 下真实 909M
  aggregator 有 **68.5%（≈623M）参数从不更新**，fp32 下 0%。票面「上游是冻结场景所以无害」
  的假设**被证伪**——上游 AnySplat 自己四个 config 全是 `freeze_backbone: false`，同样中招；
  救了本仓库的是 `instseg_anysplat.yaml` 自带的 `freeze_backbone: true`。
  **硬依赖：T7 开 `float32` 之前必须先落 T12 的 ZeRO-1**（静态项 +7.27 GB）。
- [40G 显存装不装得下](tickets/R3-40G显存预算.md)：**可行，但不是现在这份代码**。8 视角 @448
  全量微调峰值 47–52 GB，缺口 9–14 GB；**给四个 DPT 栈加梯度检查点**（现仅 aggregator 与
  DINOv2 开了）后降到 22–26 GB。步时 2.0–3.5 s，10k steps ≈ 5–8 小时。

## Not yet specified

- **训练不收敛时的应对**：等 T9 smoke run 的曲线出来才知道要不要动 warmup / margin / λ_ins。
  R1 已把 margin 的**可调方向**说清（δ_pull/δ_cross 是区间内取点，δ_push 有独立佐证不要先动），
  但「该不该调」仍要曲线才能判。
- **`voxel_size` 要不要调**：现在是 AnySplat 默认 `0.002`，论文没给这个数。体素粒度直接决定
  实例特征的空间分辨率（太粗会糊掉小物体的实例边界），但要有渲染结果才能判断。
  R3 补了一条参考：`0.002` 在 VGGT canonical 尺度下 ≈ 5–6 mm，与 448px 的像素足迹同量级。
- **T12 做完后是否还需要降视角数**：R3 给的是估算区间，`voxelize_ratio` 是其中不确定度最大
  的一项。T8 会带回真实锚点、T9 会给训练峰值——**在那之前不预先决定**砍视角数还是砍分辨率。

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
- **修 `src/model/arch/iggt.py:80` 的同款 bf16 cast**（与 T11 修的是同一个 bug）：
  地图的载体 arch 锁定 `anysplat`，IGGT 不在本 effort 的路线上，改了没有验证途径。
  修法照抄 [T11](tickets/T11-修复aggregator-bf16强制转换.md) 即可——谁要训 IGGT 的
  backbone，先回去看那张票。

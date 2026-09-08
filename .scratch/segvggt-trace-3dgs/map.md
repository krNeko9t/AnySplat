# Map: 把 phys-SegVGGT 的 query 落到已有的 3DGS 上

Label: wayfinder:map

## Destination

输入**一个已经建好的高斯场景 + 它自己的相机**，跑 `arm_b_lora` 的推理，把 SegVGGT 的
128 维实例特征 trace 到高斯上，再用 object query 在 3D 里做 class-agnostic 分割，
每个实例挂上 (E, ν, ρ) —— 产出**一张能进 ICLR 2027 论文的定性图 + 一份可复现脚本**。

**判据不是"物性准"**（[phys-on-query 01 号票](../phys-on-query/issues/01-unfreeze-lora.md)
已实测物性只比常数基线好 3–13%、ν 还输给类别查表）。判据是**链路跑通且出得了图**；
3dovs 上顺路能报的 class-agnostic 3D 分割指标算捎带，不算终点。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）：ICLR 2027 截止 **2026-09-25**，
17 天。票可以是动手的。

**领域**：前馈多视角场景理解 → 已有高斯场景的 3D 实例分割 + 实例级物性。
**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

**环境**（继承 [phys-on-query 图](../phys-on-query/map.md)，不重新核）：
GPU 服务器 `bms-39468022-001`（8×A100-40G），conda env `anysplat`，
⚠️ **跑任何 python 都要带 `PYTHONNOUSERSITE=1`**（user-site 里有 torch 2.7.1+cu118 会顶掉 env 的 2.4.1+cu124）。
两个 rasterizer 都已装：`diff_gaussian_rasterization`（3DGS）和 `diff_surfel_rasterization`（2DGS）。

⚠️ **`configs/trace/*.json` 里的场景路径全部失效**：指向 `/mnt/shared-storage-gpfs2`，
本机不存在。本机实有的高斯场景只有两组（见 03/08 号票）。

### 本图开工前已定的事（2026-09-08 grilling，不再重开）

1. **终点 = 定性图 + 可复现脚本**，量化只捎带 3dovs 的分割指标；不承诺物性的任何量化声明。
2. **trace 的是 128 维 feature field，分 7 趟 ×20 通道**，不重编译 `TRACE_CHANNELS`。
3. **场景**：主用 `3dovs/bench`（2DGS，自带 `id_maps` GT），另加 `mipnerf360/garden` 证明 3DGS 后端也通。
4. **只用 `arm_b_lora`**（`/mnt/storage_pool/liaoyuanjun/runs/exp_phys_query_arm_b_lora/2026-09-07_17-09-03/checkpoints/epoch_114-step_20000.ckpt`）。
   分割是这条链路上唯一站得住的东西，b 臂 120/120 全胜；物性两臂半斤八两，不值得多跑一遍。
5. **一致性判据用定性 + 端到端旁证**，不定硬阈值（现在没有依据，容易定出个假精确）。

### 三条把方案钉死的事实（2026-09-08 查证，是本图的地基）

**F1 — feature map 是 128 维。** `aggregator.py:215` 是 `instance_queries_proj = nn.Linear(instance_query_dim, 128)`，
mask 由 `einsum("bqd,bshwd->bqshw", proj(q), feat)` 得到（`segvggt.py:163`）。
与 `TRACE_CHANNELS = 20`（`src/trace_render/trace_rasterize.py:20`）差 6.4×，不是"略超"。

**F2 — 物性不走 mask，也不走 feature map。** `query_phys_mu/var = QueryPhysGMReadout(query_embed)`
（`segvggt.py:183`），输入是 `[1,400,1024]` 的 query 向量本身，输出 `[1,400,3]`。
它在解码之前就全算好了。选出实例后物性就是**按 `query_idx` 取行**再 `physgm_denormalize`。
`scripts/segvggt_infer.py:137` 的 `report_physics` 已实现这一整段。
⇒ **trace 只负责搬"谁在哪"，物性一个字节都不用搬。**

**F3 — 交付的 ckpt 不预测类别。** joint 配方 `class_agnostic: true`，
`loss_segvggt.py:274-277` 把 200+1 的头 logsumexp 成 2 列、`:194-196` 把 GT 类别全填 0。
⇒ 只有 objectness，没有 class。（继承自 phys-on-query 图的 Out of scope。）

### 本图的核心论证：为什么是 trace-first，不是 decode-first

Q=400 是**张量的一个轴**，不是循环次数——一次 einsum 出全部 400×V×h×w 的 mask logit。
"要 query 400 次"这个担心不存在。真正的问题是**分批**。

训练分布被 [phys-on-query 05 号票](../phys-on-query/issues/05-training-operating-point.md)
钉死在 **4 视角 / 252×448**，而一个高斯场景有几十上百个视角 ⇒ 必须分批 forward。
DETR 式 slot **身份不跨批**：批 1 的 37 号 query 和批 2 的 37 号 query 不是同一个物体。

- **decode-then-trace 在分批下产不出完整实例**：批 b 解码出的实例，mask logit 只存在于批 b 那 4 个视角上，
  trace 只能标记那 4 个视角看得见的高斯，其余高斯永远空白 ⇒ 每个实例都是碎片。
- **trace-first 没这个问题**：`gau_feat` 是全部视角累加的，`gau_feat @ q` 对**每一个**高斯都给得出 logit，
  包括这个 query 自己那批从没见过的高斯。**这是质的差别，不是省钱。**
- **成本也是 trace-first 赢**：decode-first 的通道数 M = Σ_b N_b（25 批 × ~15 ≈ 375）⇒ 19 趟；
  trace-first 固定 128 ⇒ 7 趟，且可复用。

**等价性**（两条路在同一批视角内逐字相等，差别只在覆盖）：trace 是 α·T 加权累加，
mask logit 是 `q·f`，点积线性 ⇒ `Σ αT·(q·f) = q·(Σ αT·f)`。

**残留问题 = "用哪批的 query"**：一批 4 视角只含那 4 个视角里的物体，单批 query 不覆盖场景。
做法是**把所有批的存活 query 并成 `Q_all [M,128]`**（每行自带 score 和它那行物性），
`logits = gau_feat @ Q_all.T → [P,M]`，再**在 3D 里按高斯集合 IoU 去重合并**。
合并确实回来了，但合并的是 M≈几百个已成形的物体假设，不是在 10⁶ 个点上做特征空间聚类。

**⚠️ 反噬**：trace-first 对跨批一致性的要求**比 decode-first 更严**——不同批的特征被平均进
同一个高斯，一旦漂移，平均出来是糊的；decode-first 至少每批内部自洽。
⇒ [01 号票](issues/01-cross-batch-feature-consistency.md)是地基，阻塞下游全部。

**IGGT 的现状（对照）**：`scripts/trace_instance_to_gaussians.py:472` 是个裸循环，
`encoder_batch_size` 默认 4、配置里写 48，所有批的特征进同一个 `sum_gau_sem` 和同一次全局 HDBSCAN。
**仓库已经在依赖跨批一致性，且从没检查过。** IGGT 的特征直接进对比损失、逐像素、训练时喂随机视角子集，
所以隐式被正则了；SegVGGT 的 feature map **只通过 `q·f` 被监督**，损失看不见 `f` 本身，
唯一的锚是 query 从固定的 `instance_query_token (1,400,1024)` 出发 —— **锚更弱，所以必须实测。**

## Decisions so far

<!-- 一行一个已关闭的票 -->

- [01 — SegVGGT 的 feature map 跨 forward 漂不漂](issues/01-cross-batch-feature-consistency.md)：
  **不漂，放行 trace-first。** 跨批逐像素余弦单峰（p05 0.991 / p50 0.999），控制组（同批两遍）
  恒等给了噪声底；IGGT 同装置对照更松（p05 0.982）⇒ SegVGGT 比仓库已在依赖的东西还紧。
  端到端旁证：query 打到平均特征上 mask IoU 中位 0.937，**比打到单个别的批还高**，平均在去噪不在抹糊。
  三条带下游：① query 比 feature 漂得多 ⇒ 04 号票必须在 3D 里按高斯 IoU 去重，不能用 query 余弦；
  ② 换 slot 只动尺度不动方向，但有 3% 的 norm 地板 ⇒ 04 定阈值别比它细；③ 只测了 bench 一个场景，08 顺手复跑。

## Not yet specified

- **没被任何 query 认领的高斯**（背景、漏检、只被一两个视角扫到的）怎么处理：
  丢掉？归第 0 类？还是当作 recall 缺口报出来？等 04 号票出图看比例才谈得上。
- **物性在 3D 空间的粒度**：现在物性挂在实例上是一个常数（一个 query 一行）。
  真要做 per-gaussian 的物性场（同一物体内部材质渐变），表示就得换。
  与 phys-on-query 图 Not yet specified 里的 **part-level 粒度**是同一件事的两端，
  但在本图里连问题都还没成形。
- **从旧机取回 PhysicsDreamer / alocasia 那批场景**：那是唯一"本身就是物性研究对象"的高斯场景，
  拿它出图比 3dovs 有说服力得多。但数据在已失联的 `/mnt/shared-storage-gpfs2` 上，
  是重传还是重建（跑一遍 3DGS 训练）没估过成本 ⇒ 现在不够格开票。
- **128 维 field 的复用价值**：既然场建出来了，理论上可以交互式地喂新 query（甚至文本 query）反复问。
  属 demo 层面，deadline 之后再说。

## Out of scope

- **类别预测**：继承 [phys-on-query 图](../phys-on-query/map.md)的判定——
  joint 配方 `class_agnostic: true`，本条链路上的 checkpoint 只预测 objectness。
  本图**只做 class-agnostic**，"class object" 那一半在这条路上不存在。
- **重训 / 改物性头 / 改损失**：本图只**消费**已交付的 `arm_b_lora` ckpt。
  物性烂在哪、要不要改头，归 [phys-on-query 14 号票](../phys-on-query/issues/14-checkpoint-diagnosis.md)，不归本图。
- **重编译 `TRACE_CHANNELS = 128`**：2026-09-08 已判 —— 分 7 趟是纯 Python 侧改动，
  换掉一次要动两个 CUDA 扩展、且 kernel 里 per-pixel 累加数组的寄存器压力未知的重编译，划算。
  （若 02 号票实测 7 趟慢到不可接受，这条可以重开，但要带实测数字回来。）
- **`arm_a_frozen`**：2026-09-08 已判只用 b 臂。
- **修 IGGT 的跨批一致性**：01 号票会顺带量一下 IGGT 作为对照，但**只量、不修**。
  IGGT 路线本身已被 phys-on-query 图判出 scope。
- **重跑伪标签 / 抬高物性上限**：继承 phys-on-query 图，一定会做，不在这张图里。

# 01 — SegVGGT 的 feature map 跨 forward 漂不漂

Type: prototype
Status: closed (2026-09-08) — 放行 trace-first
Blocked by: —
Blocks: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Assignee: claude (wayfinder session, 2026-09-08)

> **这是本图的地基票。** trace-first 整条路线建立在一个从没有人量过的假设上：
> 同一个物体在两次不同的 forward 里，`semantic_feature_maps` 落在同一个坐标系里。
> 假设不成立，`sum_gau_sem` 里平均的就是糊的东西，而且**不会报错**，只会出一张烂图。

## Question

`semantic_feature_maps`（`[B,V,h,w,128]`）**只通过 `q·f` 被监督**——
`loss_segvggt` 看得见的是内积，看不见 `f` 本身。原则上 `(q, f)` 可以每批一起转一个正交阵，
损失一分不变。唯一的锚是 query 从固定参数 `instance_query_token (1,400,1024)` 出发、
过固定的 `instance_queries_proj`（`aggregator.py:184,215`）。**有锚，但是间接的。**

对照：IGGT 的特征直接进对比损失、逐像素、训练时喂随机视角子集 ⇒ 跨批不变性被隐式正则。
而 `scripts/trace_instance_to_gaussians.py:472` 已经在裸循环分批跑 IGGT 并把所有批
丢进同一个累加器 —— **仓库已在依赖这条性质，从没检查过。**

要回答的是：**SegVGGT 的 feature map 跨批漂移，是否小到可以把不同批的特征平均进同一个高斯。**

## 怎么做

场景 `3dovs/bench`（`/home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench/images`）。
ckpt: `arm_b_lora` 的 `epoch_114-step_20000.ckpt`。

1. 选一个视角 `v`，让它分别落进**两个成分不同的 4 视角批**（例如 `[v,a,b,c]` 和 `[v,d,e,f]`）。
2. 比两次 forward 在 `v` 上的 `semantic_feature_maps`：**逐像素余弦相似度的直方图**。
   （先各自 L2 归一化；顺便记 norm 的比值，能分辨"转了"和"缩放了"。）
3. 同一对批次，比两边**存活 query 的 embedding**：`Q_1 [N1,128]` vs `Q_2 [N2,128]`，
   看匈牙利匹配后的余弦——query 漂不漂和 feature 漂不漂是同一枚硬币的两面。
4. **端到端旁证**（判据的另一半）：拿同一个 query，分别在「只用一批视角 trace 出的场」和
   「用全部视角 trace 出的场」上出一次 3D mask，看后者是不是塌了。
5. **顺带对照**：同一套流程量一遍 IGGT（`--feature_source iggt`），看它的直方图长什么样。
   给 SegVGGT 的数字一个横坐标。**只量，不修。**

## 判据（2026-09-08 grilling 已定）

**定性直方图 + 端到端旁证，不定硬阈值。** 现在没有依据支撑任何一个具体阈值，
定出来是假精确。主峰靠近 1 且端到端不塌 ⇒ 放行 trace-first；
明显双峰或主峰偏低 ⇒ 走 Not yet specified 里的退路（单次 forward 吃全部视角）。

## 交付

一个一次性脚本（可以扔在 `.scratch/` 下，不进 `src/`）+ 直方图图片 + 本票的 resolution 里
写下读数和放行/不放行的判断。


---

## Resolution（2026-09-08）

**放行 trace-first。** SegVGGT 的 feature map 跨 forward 不漂——至少漂得比
仓库已经在依赖的 IGGT 还少。

### 装置

`.scratch/segvggt-trace-3dgs/prototypes/measure_cross_batch_drift.py`
（一次性脚本，捕获在 `research/01-cross-batch-feature-consistency` 分支，commit `f567bc7`；
主干不留）。场景 `3dovs/bench` 36 视角，ckpt `arm_b_lora epoch_114-step_20000`，
**252×448 / 4 视角**（[05 号票](../../phys-on-query/issues/05-training-operating-point.md)钉死的训练工作点）。
4 个 anchor × 3 个成分互不相交的批，anchor 恒放 slot 0，只比 anchor 自己那张 feature map。
图：`../prototypes/out/drift.png`，全部数字：`../prototypes/out/report.json`。

**装置里最要紧的一件事是控制组**：同一批跑两遍。没有噪声底，0.99 的余弦读不出意思。

### 读数

| 量 | p05 | p50 | mean | min | n |
|---|---|---|---|---|---|
| **控制组** 同批两遍，feature cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 113k |
| 跨批 feature cos | 0.9908 | 0.9991 | 0.9975 | 0.8595 | 226k |
| **IGGT 同装置对照** | 0.9824 | 0.9992 | 0.9965 | 0.8533 | 903k |
| 跨批 feature norm 比 | 0.9396 | 1.0054 | 1.0063 | 0.7903 | 226k |
| 跨批 query cos（同 slot 号） | 0.8847 | 0.9842 | 0.9643 | 0.2589 | 3200 |
| 跨批 query cos（匈牙利，仅存活） | 0.6262 | 0.9717 | 0.9354 | 0.4376 | 101 |
| **mask IoU：q_k vs mean(f)** | 0.5381 | 0.9372 | 0.8886 | 0.0 | 130 |
| mask IoU：q_k vs f_j | 0.3631 | 0.8957 | 0.8217 | 0.0 | 260 |

**控制组恒等到逐位**（bf16 aggregator 也确定性），所以上表每一位小数都是真漂移，不是采样噪声。

### 判据怎么落的（判据在票里已定：定性直方图 + 端到端旁证，不设硬阈值）

1. **单峰，没有第二个峰。** cos 直方图（左图，对数纵轴）质量全压在 1 附近，
   0.85 以下一个点都没有。"每批转一个正交阵"那个担心**没有发生**。
2. **横坐标给了答案。** IGGT 在同一套装置下 p05 = 0.9824，SegVGGT 是 0.9908 —— **SegVGGT 更紧**。
   `trace_instance_to_gaussians.py:472` 那个裸循环已经在生产里依赖这条性质，
   而 SegVGGT 在它之上。"锚更弱所以必须实测"这个担心是对的，实测的结论是锚够用。
3. **端到端旁证没塌。** `mean(f)` 是 `sum_gau_sem/num_ray` 的 2D 替身
   （后者同样是**原始未归一化**特征的平均，`trace_instance_to_gaussians.py:1562`）。
   拿 q_k 打到 mean(f) 上，mask IoU 中位数 **0.937**，且**比打到单个别的批 f_j 上还高**
   （0.937 > 0.896）—— 平均起的是去噪作用，不是抹糊。这正是 trace-first 需要的方向。

⇒ **Not yet specified 里"单次 forward 吃下全部视角"的退路，不必走。**

### 三条带进下游的注意事项

1. **query 比 feature 漂得多**（同 slot p05 0.885，min 0.26；匈牙利 p05 0.626）。
   跟 F2/F1 一致：损失只看 `q·f`，(q,f) 可以互相补偿。
   ⇒ **不要用 query 余弦去跨批合并实例**，
   [04 号票](04-query-pooling-and-3d-dedup.md)按计划**在 3D 里用高斯集合 IoU 去重**是对的选择。
   这条把 04 的方案从"一个选项"抬成"唯一站得住的选项"。
2. **换 slot 只动尺度，不动方向。** 同一批视角、anchor 从 slot 0 挪到别处：
   cos p05 **0.9974**（比跨批更紧），但 norm 比中位数 **0.9695** —— 非 slot-0 的特征
   系统性大约 3%。VGGT 拿第一帧当参考系，这个不对称是结构性的。
   方向不变 ⇒ 对 `q·f` 的 argmax 无害；但 `sum_gau_sem` 是**原始特征**的平均，
   同一个高斯若在不同批里落在不同 slot，权重会差 3%。
   ⇒ 04 号票定阈值时记着有这么个 3% 的地板，别把阈值卡到比它还细。
3. **上面全部只测了一个场景**（`3dovs/bench`，室内、2DGS）。
   [08 号票](08-3dgs-backend-on-garden.md)跑 `mipnerf360/garden` 时顺手复跑一次这个脚本
   （`--image_dir` 换掉即可），室外/3DGS 那边免费拿到第二个读数。

### 反向没证的事

**mask IoU 有 0.0 的尾巴**（两个 IoU 列 min 都是 0）。低分存活 query（score 刚过 0.25）
本来就不稳，换个批就换个物体。这不是坐标系问题，是**弱实例假设**问题，
归 [04 号票](04-query-pooling-and-3d-dedup.md)的 score 门限去处理，不影响本票的判定。

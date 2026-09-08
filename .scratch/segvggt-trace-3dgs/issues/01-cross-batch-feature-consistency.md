# 01 — SegVGGT 的 feature map 跨 forward 漂不漂

Type: prototype
Status: open
Blocked by: —
Blocks: [04 — 跨批 query 池化 + 3D IoU 去重](04-query-pooling-and-3d-dedup.md)
Assignee: —

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

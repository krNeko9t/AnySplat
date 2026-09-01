---
id: R2
title: 跨视角 instance id 是否真的对齐
type: wayfinder:research
status: closed
assignee: krNeko9t (research agent)
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

## 解决

findings（每条 claim 带 `文件:行号` 出处）：
[`.scratch/instancesplat/research/R2-跨视角id是否对齐.md`](../research/R2-跨视角id是否对齐.md)

### 两个子集各自的结论

| 子集 | 结论 | 证据强度 |
|---|---|---|
| `processed_scannetpp_v2` | **场景级全局一致** | **强**：id 锚定在与视角无关的 3D 标注上（IGGT 论文 §2，外部来源），manifest 按文件名 stem 配对不可能错位（`scripts/make_manifest_inscene_scannetpp_v2.py:83-88,103`），且仓库已在该子集上实训通过（commit `b486ef9`）。但**从未被任何一次跨视角 loss 间接检验过**（`multi_view` 只在 infinigen 上开过），故不写"确证"。 |
| `processed_re10k` | **构造上全局一致（SAM2 track id），残余漂移未量化**——**不是**「逐帧独立」 | **弱–中**：只有 IGGT 论文一句 "each instance retains a unique ID across all views"，无任何量化；且本仓库该子集的加载路径**当前是坏的**。 |

票里"re10k 可能逐帧独立"的猜测**被证伪**。但"一致性弱于 scannetpp_v2"成立，弱点很具体：
tracker 无 3D 锚点，有两类失效——**碎裂**（同物体两 id，使 Eq.7 在该物体上变 no-op，**无害**）
与**漂移致 id 碰撞**（同 id 两物体，**会主动拉近不同物体的原型**，即票里担心的有害情形）。
论文的双向传播与关键帧重播种主要针对碎裂；漂移被承认存在但残余率未给。

**顺带查实的两条（Q1/Q3/Q4 的直接答案）：**

- **两个脚本都不生成 id**，只是路径索引器，id 语义 100% 继承自上游 `refined_ins_ids/*.npy`
  （`make_manifest_inscene_scannetpp_v2.py:54,135` / `make_manifest_inscene_re10k.py:36,86`）。
- **`src/dataset/` 全链路对 id 零重映射**：读盘（`dataset_manifest.py:150-157`）、
  最近邻缩放（`:224-229`）、`rescale_mask` 的 `cv2.INTER_NEAREST`（`crop_shim.py:37-57`）、
  翻转增强（`augmentation_shim.py:27-28`）、`default_collate`（`collate.py:19`）——
  **票里"最隐蔽的坑"不存在**，上游有多少一致性就有多少进到 batch 张量。
- **`instance_valid_mask` 在两个子集上都覆盖 100% 像素**——因为它**从未被生产**：三个
  wrapper 都按 I1 的结论拒绝把深度 `valid_mask` 转成它
  （`iggt_wrapper.py:110` / `anysplat_wrapper.py:178-179` / `segvggt_wrapper.py:227-228`），
  loss 侧 `get("instance_valid_mask")` 恒为 `None`（`loss_disc.py:191` 等）。
  re10k 更是连深度都没有（`dataset_manifest.py:239-240,261-262`）。
- **id 0 语义未测定**，且**不阻塞本 effort**：Eq.4–7 是对比式的，id 0 整体不参与、不产生梯度
  （既有实现 `loss_disc.py:112`、`loss_mvc.py:323`；同 `docs/repo_knowledge.md:108` 的结论）。
  它只对 SegVGGT 那条集合预测线要紧，而那条线在 Out of scope。
- **【新发现·代码事实】`scripts/make_manifest_inscene_re10k.py:85-86` 写出的路径缺场景目录**
  （scannetpp 有 `scene_rel` 前缀，`:132-135`），而 `DatasetManifest.root` 是整库唯一
  （`dataset_manifest.py:96,134-136`）。**re10k 现在根本加载不了**。`git log` 的 `b486ef9`
  提交信息原文即"…单独 ScanNet++ v2 数据集训练验证通过。**re10k未定**"。

### 建议：选 (a)，用 per-sample 门控落地

**政策 = (a) 只在 `scannetpp_v2` 上开 $L_{cross}$；机制 = (c) 的 per-sample 开关。**
两者不冲突：(c) 只是 (a) 的实现手段，因为一个 batch 会混着两个子集的样本。

- **不选 (b)「整个 re10k 不用」**：Eq.5 pull / Eq.6 push 都是**视角内**的，完全不依赖跨视角
  id 对齐；re10k 仍能对 $L_{pull}/L_{push}/L_{rgb}/L_{bd\text{-}rgb}$ 与几何蒸馏贡献全量
  信号。只关 Eq.7 是**损失最小**的处置。丢掉 5,137 个场景去换一个**尚未量化**的疑虑，不划算。
- **不选"数据驱动的纯 (c)"**：那需要一个 per-pair 的 id 质量估计量，仓库里没有，且会引入
  论文没有的超参——地图明令禁止。
- **门控零侵入**：`merge_manifests_inscene15k.py:33-38` 已打 `spp_` / `re10k_` 前缀 →
  `dataset_manifest.py:282` 写进 `example["scene"]` → `types.py:33` 声明 `scene: list[str]`
  （`default_collate` 原样保留）→ 每个 Loss 的 `forward` 都拿得到 `batch`（`loss.py:29-36`）。
  **在 loss 层内部读前缀即可，dataset 层一行不动。**

### T2 需要相应改什么

1. **Eq.7 变成有条件项**：`LossInsGroundCfg` 加门控字段（如
   `cross_view_scene_prefixes: list[str] | None`，默认只放行 `spp_`）。这是**对论文的有意
   偏离**（论文假设 GT id 已跨视角对齐，我们对 re10k 无法确认该前提），按地图规矩必须写进
   T2 的 `## 解决`。
2. **L_cross 逐 batch item 算，门控也逐 batch item 判**——batch 内两子集混排是常态，不能整批
   开关。照抄 `loss_disc.py:243-252` 的 `for b in range(B)`（这同时也是 id 只在场景内唯一
   所要求的：跨 B 汇总会把不同场景的同号 id 当成同一物体）。
3. **新增一条边界情形**：整批全是 re10k → L_cross 恒 0，但**必须留在 autograd 图上**
   （与已列的"无有效实例"同一个坑，`fix(I3)`）。
4. **valid 判据要容忍 `instance_valid_mask is None`**：T2 现写的三条件并列，但按上文该字段
   今天恒为 `None`；若实现成 `None → 全 False`，loss 会静默恒 0。必须显式 `None → 全 True`。
5. **测试补两条**：① 双视角、同物体在两帧带不同 id → cross 应为 0（非 NaN、无虚假拉近）；
   ② 门控测试：`re10k_` 样本 cross=0、`spp_` 样本 cross>0、混合 batch 两者并存。
6. **T2 不负责修 re10k 路径 bug**（建议另开票，归 T7 / 数据准备那条线）。但在它修好前
   re10k 一帧都进不了训练，**地图的数据配比短期内事实上就是"只有 scannetpp_v2"**。

### 待办（不阻塞 T2）

一次性验证脚本（**本票不写**，findings §5 给了设计）：纯 numpy 读 `refined_ins_ids/*.npy` +
rgb，无需深度/GPU/模型，三层判据 = ① id×frame 占据矩阵判"是不是全局"（决定性）；
② 碎裂率；③ **碰撞率**（同 id 在两帧的平均 RGB / 面积 / 质心位移描述子距离）——③ 才是要看
的数。scannetpp_v2 上同脚本跑一遍作对照基线。若 re10k 的③足够低，**改一个配置项**即可翻转
门控放行，不用改代码。顺手把 `scripts/check_instance_id0.py` 在两子集各跑一次结掉 id-0 语义
（服务 SegVGGT 线）。前提：先修 §1.3 的路径 bug，并排除 `zip(sorted(...))` 位置配对错位
（`make_manifest_inscene_re10k.py:46-48,59`）——错位的表现恰恰就是"跨视角 id 全对不上"，
别把自己的 bug 归因给 IGGT。

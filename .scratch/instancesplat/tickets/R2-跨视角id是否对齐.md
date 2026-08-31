---
id: R2
title: 跨视角 instance id 是否真的对齐
type: wayfinder:research
status: closed
assignee: codebuddy-main
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

**结论：`scannetpp_v2` = 全局一致；`re10k` = 部分一致（未验证的假定一致）。
建议 = (a)+(c) 的合体：per-subset 开关，$L_{cross}$ 只在 `scannetpp_v2` 样本上启用；
`re10k` 保留在数据配比里，但只参与 $L_{pull}/L_{push}$（视角内项不受跨帧 id 影响）。**

1. **manifest 脚本只透传路径，不读 mask**：两子集的 mask 都来自上游 `refined_ins_ids/` 产物。
   `scannetpp_v2` 按 stem 建索引、帧对齐靠 metadata（`scripts/make_manifest_inscene_scannetpp_v2.py:82-88,135`）；
   `re10k` 靠排序 zip 配对、只校验数量，docstring 自述目录结构是「假设」
   （`scripts/make_manifest_inscene_re10k.py:13-20,46-59,86`）。脚本既不保证也不破坏一致性。
2. **读取链路零重映射**：dataset → shims → collate → wrapper 对 id 只有最近邻 resize / 中心裁剪 /
   水平翻转（`src/dataset/dataset_manifest.py:150-157,224-229`、`shims/crop_shim.py:196-207`、
   `shims/augmentation_shim.py:27-30`），全局 id 不会被打散；但上游若不一致也**不会被修复**。
   且 loss 侧全部按「跨帧同 id」消费：`loss_disc.py:248-258` 跨视角合并求实例均值、
   `loss_segvggt.py:162` 跨帧 `torch.unique`——任何不一致会被**静默合并成同一实例**，
   正好坐实票里担心的「主动把不同物体拉到一起」。
3. **re10k 猜测证实**：它是 IGGT 用 SAM + tracking 生成并 refine 的伪标注
   （`ref_knowledge/InstanceSplat/InstanceSplat.md:192`）——track 内跨帧一致，但天然存在
   track 断裂 / id 切换风险。仓库内**没有任何**跨帧一致性生成/校验/修复代码
   （`check_instance_id0.py` 逐帧查 id-0 形状；`check_i1_valid_mask_vs_instance.py:138-139`
   对无深度的 re10k 直接退出）。`docs/repo_knowledge.md:109` 的「多视角一致 id」是消费侧假设，
   不是验证结论。
4. **`instance_valid_mask` 与 id 0**：valid mask 当前**没有任何模块生成**（默认 None = 全有效；
   三个 wrapper 显式拒绝用 depth valid_mask 复用，`iggt_wrapper.py:110`、`segvggt_wrapper.py:227-228`、
   `anysplat_wrapper.py:178-179`，历史问题 I1）。id 0 = background/ignore/invalid
   （`docs/conventions.md:9`）；是「真背景」还是「未标注」需逐源实测——`check_instance_id0.py`
   的判据 = id-0 区域中**不触碰图像边界的紧致连通域**像素占比（< 0.02 真背景，> 0.08 藏未标注物体，
   L146-156）。
5. **对 T2 / 数据配比的影响**：T2 实现 $L_{cross}$ 时需要按数据源 gating——manifest 合并时已带
   `scene_id` 前缀（`spp_` / `re10k_`，`scripts/merge_manifests_inscene15k.py:33-38`），可零成本判来源。
   在实测出数之前，`re10k` 样本一律关掉 $L_{cross}$。实测脚本另立新票 **T11**。

---
id: T2
title: 原型式实例 grounding loss
type: wayfinder:task
status: open
assignee: -
blocked-by: [R1, R2, T1]
---

## Question

实现论文 3.2 的 $L_{ins} = \lambda_{pull}L_{pull} + \lambda_{push}L_{push} + \lambda_{cross}L_{cross}$
（Eq.4–8），作用在 **T1 渲染出来的** $\tilde{S}_i$ 上，不是在 encoder 的 2D 特征图上。

新文件 `src/loss/loss_ins_ground.py`，注册名建议 `ins_ground`。**不要碰 `loss_disc.py`**——
那是既有路线在用的，本 loss 与它并存、互不引用。

公式要点（照抄论文，别自由发挥）：

- Eq.4：逐 valid 像素 $\ell_2$ 归一化，再按 GT 实例求均值得原型 $\bar{f}_k^{(i)}$。
  **注意原型是「先归一化再平均」，平均后不再归一化**（Eq.14 的语义原型才有外层 norm）。
- Eq.5 pull：$\mathbb{E}_p[\|f_i(p) - \bar{f}_k^{(i)}\|_2 - \delta_{pull}]_+$，先对实例内像素
  取期望、再对实例取平均。
- Eq.6 push：**视角内**实例原型两两之间，$[\delta_{push} - \|\bar{f}_k - \bar{f}_l\|_2]_+$，
  带权 $\eta_{k,l}^{(i)}$。**本期 $\eta \equiv 1$**（语义分支在 Out of scope，论文说
  "$\eta=1$ for uniform separation"）。但把 $\eta$ 留成可传入参数，将来接语义不用改公式。
- Eq.7 cross：**同场景的视角对** $i<j$、共同可见的实例 $k$，对齐两个视角各自的原型。
  依赖 R2 的结论——如果某子集跨帧 id 不一致，这一项在该子集上必须关掉。
- margin 三个值来自 R1。
- $L_{push}$ 和 $L_{pull}$ 是**对所有监督视角求平均**后才加权。

对齐与边界条件（这些是最容易写错的地方，逐条测）：

- 哪些像素算 valid：GT `instance_mask != ignore_id` **且** `instance_valid_mask` 为真
  **且** 渲染 alpha 达标（T1 的决策）。三个条件要一致地用在原型、pull、push 上。
- 一个视角里只有 0 或 1 个实例 → push 无配对，该项应为 0 而不是 NaN。
- 一个实例只在单个视角可见 → 不进 cross 项。
- 整个 batch 没有任何有效实例 → loss 为 0 但**必须留在 autograd 图上**
  （`git log` 里 `fix(I3)` 那个坑，别再踩一次）。

**层归属**：纯公式，无可学习参数，`resolve_*` 负责 pred↔GT 对齐 —— 严格照
`docs/layered_scheme.md` 的 Loss 层契约。

**验证**（CPU 合成张量）：手算小例子对拍；同实例特征完全相同时 pull=0；
两原型距离 > $\delta_{push}$ 时 push=0；跨视角原型相同时 cross=0；梯度回到 `PartHead`；
上面四条边界条件各一个判定性测试。

## R1 / R2 关闭后追加的实现约束

margin 取值见 [R1](R1-margin-取值.md) 的 `## 解决`，$L_{cross}$ 门控政策见
[R2](R2-跨视角id是否对齐.md) 的 `## 解决`。**下面只列这两张票新引出的、写代码时最容易踩的坑**，
数值和政策本身不在这里重复。

1. **Eq.6 的 margin 不乘 2——这是最容易静默写错的一条。**
   De Brabandere 原式是 $[2\delta_d - d]_+$，论文 Eq.6 是 $[\delta_{push} - d]_+$（已核对
   `ref_knowledge/InstanceSplat/InstanceSplat.md` 的 Eq.6 原文）。照搬 `disc.yaml` 的
   `delta_d: 1.5` 再套 2 倍，目标间距 3.0 > 归一化嵌入的距离上限 2.0，**hinge 永远打不开、
   push 恒为正且梯度不停**。实现时直接用 $[\delta_{push} - d]_+$，不要乘 2。

2. **`instance_valid_mask` 今天恒为 `None`。** 三个 wrapper 都明确拒绝把深度 `valid_mask`
   当实例 valid mask（`anysplat_wrapper.py:178`、`segvggt_wrapper.py:227`、`iggt_wrapper.py:110`），
   loss 侧只有 `.get("instance_valid_mask")`（`loss_disc.py:191`）。
   → 票面 valid 三条件里的这一条，**必须实现成 `None → 全 True`**。
   若实现成 `None → 全 False`，loss 会**静默恒 0**，而且曲线上看起来「收敛得很好」。
   这条要有判定性测试。

3. **$L_{cross}$ 逐 batch item 门控，不能整批开关。** 一个 batch 混着两个子集是常态。
   门控零侵入路径已查实：`merge_manifests_inscene15k.py:33-38` 打 `spp_` / `re10k_` 前缀
   → `dataset_manifest.py:282` 写进 `example["scene"]` → `types.py:33` 声明 `scene: list[str]`
   （`default_collate` 原样保留）→ `loss.py:29-36` 每个 Loss 的 `forward` 都拿得到 `batch`。
   **在 loss 层内部读前缀即可，`src/dataset/` 一行不动**（合乎地图的不干涉约束）。
   配置字段建议 `cross_view_scene_prefixes: list[str] | None`，默认只放行 `spp_`。
   这是**对论文的有意偏离**（论文假设 GT id 已跨视角对齐，我们对 re10k 无法确认该前提），
   按地图规矩必须写进本票的 `## 解决`。

4. **`for b in range(B)` 是必须的，不只是为了门控。** id 只在场景内唯一，跨 batch 汇总会把
   不同场景的同号 id 当成同一物体。照 `loss_disc.py:243-252` 的写法。

5. **新增边界情形**：整批全是 `re10k_` → $L_{cross}$ 恒 0，但**必须留在 autograd 图上**
   （与票面已列的「无有效实例」是同一个坑，`fix(I3)`）。

6. **健康的 step 0 是三项全满值。** Eq.4 的原型「先归一化再平均、平均后不再归一化」，
   所以原型落在单位球**内部**，初始化时模长≈0、所有原型挤在原点。
   → 开局 pull/push/cross **任何一项读数为 0 都是 bug 信号**，不是收敛。
   这条同时是 [T5](T5-嵌入质量诊断看板.md) 的免费断言。

7. **测试补两条**：① 双视角、同物体在两帧带不同 id → cross 应为 0（非 NaN、无虚假拉近）；
   ② 门控测试：`re10k_` 样本 cross=0、`spp_` 样本 cross>0、混合 batch 两者并存。

**本票不负责修 re10k 的路径 bug**（那是 [T13](T13-修复re10k-manifest路径.md)）。
但在 T13 关掉之前 re10k 一帧都进不了训练，门控实际上暂时是空转的——**仍然要写**，
否则 T13 一关就会静默地把未验证的 id 喂进 Eq.7。

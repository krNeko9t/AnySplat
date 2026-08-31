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

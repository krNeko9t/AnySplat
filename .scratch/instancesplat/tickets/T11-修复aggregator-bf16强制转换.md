---
id: T11
title: 修复 aggregator bf16 强制转换（backbone 实为冻结）
type: wayfinder:task
status: closed
assignee: krNeko9t
blocked-by: []
---

## Question

**这张票是 R3 挖出来的、比显存更要命的问题，且它直接决定整个 effort 的前提成不成立。**

`src/model/arch/anysplat.py:124`：

```python
self.aggregator = model_full.aggregator.to(torch.bfloat16)
```

这是**永久 dtype 转换**，不是 autocast。后果链：

1. 参数是 bf16 → AdamW 的 exp_avg / exp_avg_sq 也在 bf16，参数更新 `p -= lr * ...` 在 bf16 里做。
2. bf16 尾数 8 位，相对 ULP ≈ 2^-8 ≈ 3.9e-3。典型权重量级 0.02–0.03 → 绝对 ULP ≈ 0.8e-4 ~ 1.2e-4。
3. 地图锁定 transformer lr = base 2e-4 × 0.1 = **2e-5**，AdamW 归一化后单步更新量 ≈ lr = 2e-5。
4. **2e-5 < ULP → 更新被舍入成 0。909M 参数的 backbone 很可能根本没在训。**

而地图的整个配方建立在「backbone 解冻」上（论文 Table 3 的「3DGS frozen」消融：
冻住 T-mIoU 58.86 vs 联合训 64.03）。如果这个 bug 在，我们会**以为**自己在做全量微调，
实际跑出的是被消融证明更差的那一档，且从 loss 曲线上看不出来。

顺带澄清票面外的一个误解：Lightning `precision: bf16-mixed` **只包 autocast，不会额外维护
fp32 master weights**。所以「反正 mixed precision 会兜底」的想法不成立——这里参数本身就是 bf16。

## 要做的

1. **确认**这条推理（读 optimizer 构造路径，确认 aggregator 参数确实以 bf16 进 param group；
   确认没有别处把它转回 fp32）。**不要靠推理下结论**——本机没 GPU，但 dtype 是静态可查的。
2. 决定修法并说明理由。候选：
   - (a) 直接去掉 `.to(torch.bfloat16)`，参数留 fp32，靠 `bf16-mixed` 的 autocast 省激活显存。
     **倾向这个**：这是 PyTorch/Lightning 混合精度的标准姿势，且不引入新超参。代价是参数+
     优化器状态显存上升（R3 的分项表里已经把这笔算进去了，静态 17.0 GB 就是按此估的，
     所以**不会推翻 R3 的结论**）。
   - (b) 保留 bf16 参数但外挂 fp32 master weights。复杂，且等于自己实现一遍 AMP。
   - (c) 提高 transformer lr 让更新量超过 ULP。**不可接受**——偏离论文的 lr 配方去迁就一个 bug。
3. 查清这个 cast 是 AnySplat 上游原样继承的还是本仓库加的（`git log -L124,124:src/model/arch/anysplat.py`），
   如果是上游的，说明上游场景是**冻结 backbone**（那时 bf16 无害），我们解冻后它才变成 bug。
   这条要写进 `## 解决`，因为它解释了为什么既有实验没暴露这个问题。

## 硬约束

地图的「不干涉既有实现」：`config/model/encoder/anysplat.yaml` 的既有字段语义不动。
如果修法需要开关，用**新增的、默认保持现状**的可选字段，让既有实验的行为逐比特不变。
SegVGGT 那条线（`segvggt.py` 等）一律不碰。

## 验证（CPU）

- 构造 encoder 后打印 aggregator 各 param 的 dtype，确认为期望值。
- 构造 optimizer，取 transformer 组里一个参数，手动跑一步 `optimizer.step()`（用合成梯度，
  量级设成 2e-5），**断言参数确实变了**（`assert not torch.equal(before, after)`）。
  这个测试就是这张票的判定性证据，它在修复前应该**失败**。
- 确认既有 `instseg_anysplat.yaml` 路径的行为不变。

## 解决

**票面推理成立，而且比票里写的更糟。** 修法取候选 (a)：去掉硬编码 cast，参数留 fp32，
靠 `bf16-mixed` 的 autocast 拿算力。按硬约束包成**新增的、默认保持现状**的可选字段。

### 一、确认（不靠推理，全部静态可查 / CPU 实测）

**dtype 链路，全程没有任何 fp32 兜底：**

| 环节 | 事实 |
|---|---|
| `arch/anysplat.py:124` | `aggregator.to(torch.bfloat16)`，**永久转换**；heads（camera/depth/point）没转，仍 fp32 |
| `arch/weight_loading.py` | `load_state_dict` 是 in-place `copy_`，**保持目标 dtype** ⇒ 从 HF 加载后仍是 bf16 |
| `base_wrapper.py:597` | 裸 `torch.optim.AdamW` ⇒ `exp_avg`/`exp_avg_sq` = `zeros_like(p)` = **bf16** |
| `main.py:104` | **DDPStrategy**，不是 FSDP / DeepSpeed（`deepspeed_stage_1` 那行是注释掉的） |
| `trainer.precision` | `bf16-mixed` 在 Lightning 里 = **纯 autocast**，不转参数、不维护 master weights |

**CPU 实测**（`lr = 2e-4 × 0.1 = 2e-5`、`wd=0.05`、`betas=(0.9,0.95)`，即 `base_wrapper` 的真实超参；
在**真实的 909.1M `Aggregator`** 上跑，随机初始化，不需要 GPU 也不需要下 HF 权重）：

| | bf16 参数 | fp32 参数 |
|---|---|---|
| 50 步后**从不更新**的参数占比（5.45M 分层抽样） | **68.5%（≈623M / 909M）** | 0.0% |
| `|w| > 5.1e-3` 那批的更新覆盖率 | 12.5% | 100.0% |
| 单参数 `|w|=0.02`，单步 `torch.equal(before, after)` | `True`（没动） | `False` |

死区门槛 = `lr · 2⁸ ≈ 5.1e-3`。

**关键修正：不是「整个 backbone 冻结」，是「按权重量级选择性冻结」。**
小权重照常更新、大权重钉死，会单向压扁权重分布——比全冻结更难诊断，且 loss 曲线上完全看不出来。
另外 warmup（lr 从 `2e-8` 线性爬）和 cosine 末段（降到 `2e-6`）**两段是 100% 死**，
上面的 68.5% 是稳态最好情况。

### 二、修法

`EncoderAnySplatCfg` 新增：

```python
aggregator_param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
```

`anysplat.py` 的 cast 改由它驱动。**默认 `bfloat16` = 现状**，
`config/model/encoder/anysplat.yaml` 与 `config/experiment/instseg_anysplat.yaml`
**一个字都没改**（dacite 对有默认值的字段不要求 yaml 出现）⇒ 既有实验行为不变。
新配方在 T7 的新 experiment yaml 里显式写 `float32`。

排除另外两个候选：
- **(c) 抬 lr** —— 死区门槛是 `lr·2⁸`，要让 `|w|=0.03` 动起来得把 transformer lr 抬到 `1.2e-4`，
  等于废掉论文的 0.1× 差异化 lr。不可接受。
- **(b) 外挂 fp32 master weights** —— 重造一遍 AMP，DDP 下还要自己处理 allreduce dtype。

**白送的附带修复**：bf16 参数下 DDP 的梯度 allreduce 也在 bf16 里做（8 卡累加误差），改 fp32 一并解决。

`anysplat.py:434` 的 `image.to(torch.bfloat16)` **未动**：fp32 参数 + `autocast(bf16)` 下，
patch_embed 的 conv2d 无论如何都会被 autocast 转 bf16，改与不改逐比特等价。

### 三、票面第 3 条假设：**证伪**

`git show 8d6180e`（AnySplat 作者本人的 "release training/inference code"）就带这行 cast，
但上游自己的 `multi-dataset.yaml` / `dl3dv.yaml` / `scannetpp.yaml` / `co3d.yaml`
**全是 `freeze_backbone: false`** + `lr 2e-4 × backbone_lr_multiplier 0.1` = 同一个死区。
⇒ **上游 AnySplat 自己也中招了**，不存在「上游是冻结场景所以无害」这回事。

真正让**本仓库**既有实验没暴露问题的是另一件事：
`config/experiment/instseg_anysplat.yaml:30` 的 `freeze_backbone: true # ugly`
—— aggregator 根本没进 optimizer，bf16 参数自然无害。

### 四、验证

`.scratch/instancesplat/T11_verify.py`，本机 CPU 可跑（`paper_repo` env，torch 2.11），
不需要 GPU、不需要下 HF 权重：

```
/home/liaowanjun/miniconda3/envs/paper_repo/bin/python .scratch/instancesplat/T11_verify.py
```

四组断言全部通过：
- **[A]** ast 静态检查：字段存在、默认 `bfloat16`、硬编码 cast 已移除、两个既有 yaml 未出现该字段。
- **[B]** 真实 `Aggregator`（909.1M）在两个取值下的全部浮点参数 dtype。
- **[C]** 判定性断言：大权重那批在 bf16 下的更新覆盖率 < fp32 的一半（实测 12.5% vs 100%）。
- **[D]** 分层抽样实测死参数占比：bf16 68.5% / fp32 0.0%。

> 两条方法学修正（写下来免得后人重踩）：
> 1. 判据必须是**更新覆盖率**，不能是「整个张量 `torch.equal` 纹丝不动」——
>    bug 逐元素生效，任何真实张量都含有会动的小权重尾巴。
> 2. 解析判据 `|w| > lr·2⁸` 是**保守上界**（78.6% vs 实测 68.5%）：50 步里动量累积
>    会把边界附近的权重顶过去。要报数就报实测。
>
> 局限：随机初始化，量级与训练后的 ViT-L 同量级但非逐值相同。结论只依赖量级分布。

### 五、留给别的票的两条

1. **硬依赖 → T12**：本修法静态项 **+7.27 GB**（R3 已算过）。T7 的新 experiment yaml
   若要开 `aggregator_param_dtype: float32`，**必须先落 T12 的 ZeRO-1 优化器分片**
   （配平后静态项 11.25 GB，反而低于现状的 12.7 GB）。不配平就直接开 fp32 会撞爆显存。
2. **`src/model/arch/iggt.py:80` 有一模一样的 cast**（`base.aggregator.to(torch.bfloat16)`），
   同类 bug。**本票不修**：地图的载体 arch 锁定 `anysplat`，IGGT 不在路线上，
   顺手改会扩到没人验证的第二条线。谁要训 IGGT 的 backbone，先回来看这张票。

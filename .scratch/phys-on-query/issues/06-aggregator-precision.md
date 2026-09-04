# 06 — 909M aggregator 该用什么精度训

Type: grilling
Status: open
Blocked by: —

> **寄存票**：本票**不在本图通往终点的路上**（终点是 SegVGGT checkpoint，SegVGGT 路线
> 全程 fp32，本洞咬不到它）。它从 `.scratch/freeze-contract/map.md` 的
> [07 号票](../../freeze-contract/issues/07-bf16-silent-freeze.md) 判出 scope 后无处可去，
> 挂在这里只因为**它落在同一双眼睛前**。**它不阻塞本图任何票，也不得成为任何票的阻塞。**

## Question

`src/model/arch/anysplat.py:113` 与 `iggt.py:80` 把 909.1M 的 aggregator **永久**
`.to(torch.bfloat16)`（非 autocast）。后果不是"精度低一点"，是**按权重量级的选择性钉死**：
AdamW 的 `exp_avg`/`exp_avg_sq` 也在 bf16 里，bf16 尾数 8 位 ⇒ ULP ≈ |w|·2^-8 ⇒
量级 ~lr 的更新在 |w| > lr·2^8 时被舍成 0。小权重照常更新、大权重钉死，**单向压扁权重分布，
且 loss 曲线上看不出来**；warmup 段与 cosine 末段是 100% 死。Lightning 的 `bf16-mixed`
兜不住（只包 autocast，不维护 fp32 master weights；strategy 是 DDP 不是 FSDP，全链路无 fp32 兜底）。

**这不是冻结**：`requires_grad` 全程为真，参数进了优化器，只是权重不动——只占冻结三判据的
第三条。所以它被 freeze-contract 图判出 scope，是**训练精度策略**问题。

### 命中面（2026-09-04 按 `config/experiment/locks/` 的 22 份 lock 实测）

| 配方 | 可训 bf16 的 aggregator 参数 |
|---|---|
| `instseg_small` | **1209 个 / 909.1M**，占其 12.49 亿可训参数的 **72.8%**，`base_lr: 1e-5` |
| `co3d` / `dl3dv` / `multi-dataset` / `scannetpp` | 各 866 个（上游 AnySplat 预训练配方） |
| 其余 17 份（含全部 SegVGGT） | 0 |

`cbe93f9` 在**同一个 909.1M Aggregator** 上实测过（lr 2e-5，wd 0.05，betas (0.9,0.95)）：
**50 步后 68.5%（≈623M）参数从不更新**；fp32 下死参数 0.0%。`instseg_small` 的 lr 还小一半，
死亡率只会更高。

### 要决定的

1. `instseg_small` 到底要不要真的训这 909M——**如果要，它现在有 2/3 是假的**；如果本来就
   不指望它动，那该冻的没冻，走 `freeze_keywords`。
2. 四份上游 AnySplat 配方（`co3d`/`dl3dv`/`multi-dataset`/`scannetpp`）本仓是否真跑。
   若只是留档，不动；若要跑，同问。
3. 修法二选一：**拿掉那行无条件 cast**（改精度策略，代价是显存）／**冻掉 aggregator**
   （改配方，代价是不训主干）。**不做的是加 config 开关**——freeze-contract 07 已论证过，
   开关只提供"dtype 有没有被改过"，而 lock 的 dtype 列已经免费提供了。

## 现场事实（别重新推导）

- `git merge-base --is-ancestor cbe93f9 HEAD` → **NO**。`cbe93f9` 只在 `ft_is_cur` 上，
  是 instancesplat 图的产物，附带验证脚本 `.scratch/instancesplat/T11_verify.py`（本机 CPU 可跑，
  无需 GPU / HF 权重），本分支没有该文件。
- `iggt.py:80` 那行经 `git log -L 80,80` 溯到 `8bace2c`（"opus实现"）、`e2db80d` 搬运——
  **是本仓自己写的，上游 IGGT 并无此 cast**。照抄 AnySplat 的做法。
- `src/model/arch/segvggt.py` 干净，只用 autocast（`:150-157`），无永久 cast。
- 改动会让对应配方的 lock 的 dtype 列整体翻转 ⇒ 需按
  `.scratch/freeze-contract/` 的流程重生成并提交 lock，`git diff` 里应当只有 dtype 那一列在变。

## 完成判据

三条各有明确处置；若选"拿掉 cast"，受影响配方的 lock 已重生成且 diff 只动 dtype 列。

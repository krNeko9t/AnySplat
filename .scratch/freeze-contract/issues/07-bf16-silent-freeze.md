# 07 — 把 bf16 静默冻结的修复带到本分支

Type: task
Status: closed (out of scope)
Blocked by: —
Assignee: krNeko9t

## Question

`cbe93f9`（在 `ft_is_cur` 上）给 AnySplat 加了 `aggregator_param_dtype` 开关，修的是：
aggregator 被永久 `.to(torch.bfloat16)` ⇒ AdamW 的 `exp_avg`/`exp_avg_sq` 也是 bf16 ⇒
bf16 尾数 8 位、ULP ≈ |w|·2^-8 ⇒ **实测 50 步后 68.5%（≈623M）参数从不更新**，
且 `requires_grad=True` 全程为真、loss 曲线看不出来。Lightning 的 `bf16-mixed` 兜不住
（只包 autocast，不维护 fp32 master weights；strategy 是 DDP 不是 FSDP）。

`git merge-base --is-ancestor cbe93f9 HEAD` → **NO**。本分支上：

- `src/model/arch/anysplat.py:124` — 无条件 `.to(torch.bfloat16)`
- `src/model/arch/iggt.py:80` — 同款，`cbe93f9` 的 commit message 自陈"已记入地图
  Out of scope"（那是 instancesplat 那张图的 out of scope，不是本图的）
- `src/model/arch/segvggt.py` — **干净**，只用 autocast（`:150-157`），无永久 cast

做三件事：

1. cherry-pick / 重做 `cbe93f9` 到本分支（默认值保持上游行为，既有实验行为不变）。
2. 把同样的开关给 `iggt.py:80`。
3. 复核 `cbe93f9` 附带的验证脚本 `.scratch/instancesplat/T11_verify.py` 在本分支还能跑
   （本机 CPU 可跑，无需 GPU / HF 权重）。

**与本图的关系（2026-09-04 由 03 号票改写）**：**脱钩。** bf16 死参数不是冻结——它只占
冻结三判据的第三条（权重不变），机制在另一层，管它叫「静默冻结」是比喻。本票是**独立的洞**，
不必等指纹层落地，也不该反过来给指纹层加断言（06 号票那条 dtype 判据已整条撤销）。
指纹层对它的全部承诺只有一条：`dtype` 是指纹的一列，将来真有人给 aggregator 打开 bf16，
lock 的哈希会撞一次，人被迫看一眼。

## 完成判据

两个 arch 都有 dtype 开关且默认行为不变；验证脚本在本分支跑通并给出与 commit message
一致的数字（bf16 68.5% 死参数 / fp32 0%）。

---

## 处置（2026-09-04）：**判出 scope，整票关闭，不在本图解决**

### 决定

本票所有三件事（搬 `cbe93f9`、扩 `iggt.py`、复核 T11）**都不做**。它属于
**训练精度策略**，不是冻结机制。

### 理由

**`aggregator_param_dtype` 这个开关本身不值得**。它既不解决问题，也不发现问题——
它能告诉你的只有「dtype 有没有被改过」。而"有没有被改过"恰恰是 03 号票已经安排好的
那一列 `dtype` 免费提供的：真有人改，lock 的哈希撞一次。加一个 config 字段，是在
**已有的可见性之上再叠一层同样只有可见性的东西**，还顺带违反 Notes 第 4 条的精神
（多一个 config 能拨动的旋钮）。真正的修法是把那行无条件 `.to(torch.bfloat16)` 拿掉，
那是精度策略的改动，不是开关的改动。

**它跟本图的终点没有交集**。03 号票已经论证过一次：bf16 死参数只占冻结三判据的第三条
（权重不变），机制在另一层。本次核码没有推翻这条——推翻的只是「零次发生」那个**读数**，
读数变了不等于判据变了。这与 `lr` 出图、跨 stage 校验出图是同一条理由：
**不占三判据的东西，不由冻结层持枪站岗。**

### 现场核出的两条事实（随票留档，不随票消失）

1. **命中面比 01 号票的读数大得多**。09 落地后按 22 份 lock 统计 aggregator 参数的
   `(requires_grad, dtype)` 分布：**5/22 存在可训的 bf16 aggregator 参数**——
   `instseg_small`（1209 个 / **909.1M**，占该配方 12.49 亿可训参数的 **72.8%**，
   `base_lr: 1e-5`，比 `cbe93f9` 实测样本的 lr 还小一半）、
   `co3d` / `dl3dv` / `multi-dataset` / `scannetpp`（各 866 个）。
   01 号票「bf16 静默冻结在这九份里零次发生」是在**九份**上量的，08 把面扩到 22 份后过期。
   **本图的产物证伪了本图的事实底座**，靠的正是 03 坚持保留的那一列——「保住可见性」当场兑现。
   ⇒ 已回写 map 事实底座。
2. **`iggt.py:80` 是我们自己写的，不是 vendored 上游**。`git log -L` 溯到
   `8bace2c`（"opus实现"）首次写入、`e2db80d`（encoder 并入 arch）搬运。上游 IGGT
   **没有**把 VGGT 主干无条件转 bf16——这行是照着 AnySplat 的做法复刻的。
   所以它不是"上游的锅传染过来"，是本仓自己的精度策略选择。

### 这个洞去哪

留在 `ft_is_cur` 分支上的 `cbe93f9`（instancesplat 图的产物）不动。本分支上
`anysplat.py:113` 与 `iggt.py:80` 两行无条件 cast **原样保留**，等一次专门的
**训练精度策略**处置——那时该问的是「909M 主干该用什么精度训」，而不是「要不要加个开关」。
本图对它的全部承诺只有一条，且已兑现：`dtype` 是 lock 的一列，改了哈希会撞。

# 07 — 把 bf16 静默冻结的修复带到本分支

Type: task
Status: open
Blocked by: —

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

**与本图的关系**：这张票本身是补一个已知洞，但它同时是 03 号票"dtype 必须进指纹"
那条论据的**实物证据**——03 的指纹设计要能抓住这个 case。

## 完成判据

两个 arch 都有 dtype 开关且默认行为不变；验证脚本在本分支跑通并给出与 commit message
一致的数字（bf16 68.5% 死参数 / fp32 0%）。

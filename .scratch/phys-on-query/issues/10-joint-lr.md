# 10 — joint 配方的 lr 到底该是多少

Type: grilling
Status: open
Blocked by: —

> 由 [01 号票](01-unfreeze-lora.md) 顺带查出。**刻意不在收 01 号票时顺手改**——
> 顺手改 = 一次没人复核的配方变更，而它直接影响第一个 checkpoint 的质量。

## Question

`config/experiment/segvggt_agnostic_phys_joint.yaml` 的 lr **注释与实际不符**：

- 实际：`optimizer.lr: 1.0e-4` + `backbone_lr_multiplier: 1.0` ⇒ 默认组（`instance_` 454M +
  `semantic_head` 32.6M）**1e-4**；`param_groups[query_physgm] lr_multiplier: 5.0` ⇒ **5e-4**。
- 注释 `:44-48` 说 "randomly-init query_physgm gets a hotter group (**~1e-4**)"，
  `:65` 说 "**2e-5 × 5 = 1e-4**"。即注释描述的是 2e-5 / 1e-4 那一档。

这是 `freeze-contract` 图 01 号票的 **F1**，被那张图明文判出 scope
（"属配方审查，不由冻结层持枪站岗"，见其 Out of scope 的 `lr` 一条），因此现在不归任何图。

要定的：**哪一档是对的**——是注释过期（实际 1e-4 / 5e-4 才是想要的），
还是配置漂了（想要 2e-5 / 1e-4，被谁改成了 5 倍）。然后注释与实际对齐。

参考系：
- `segvggt_finetune_agnostic`（stage-1）为"没有任何随机初始化的新参数、是适配不是从头训"
  把论文的 `2e-4` 降到单一 **`lr: 2e-5`**（`docs/repo_knowledge.md`）。
  joint 与它的**唯一**差别是多了一个随机 init 的 `query_physgm`(~0.2M)。
  按那条推理，joint 的**预训练部分**应当同样是 2e-5，只有 `query_physgm` 该烫。
  ⇒ 现状把 454M 预训练参数一起提到 1e-4，很可能不是有意的。
- 论文 A.4：新参数 `2e-4`、pre-existing `6e-5`。

## 影响面

对 [01 号票](01-unfreeze-lora.md) 的两臂**等同施加**，因此**不影响那次对照的结论**，
但它决定两臂的绝对水平。若判定现状偏烫，两臂应在改正后的 lr 上跑。

## 完成判据

一档 lr 定下来、写进 config、注释与实际对齐；若改动了 `optimizer.lr`，
按 freeze-contract 的规矩重生成该配方的 lock（`base_lr` 在 lock 头部，不参与哈希，
但 diff 应看得见）。

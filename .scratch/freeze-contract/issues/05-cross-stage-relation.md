# 05 — 跨 stage 的冻结关系怎么表达

Type: grilling
Status: open
Blocked by: 03

## Question

单份配方自洽还不够。真正致命的是**stage 链断裂**：stage-2（`segvggt_physgm`）声称
"几何完全不动，所以这份 ckpt 应当复现 stage-1 的几何"，靠的是冻结集**覆盖了**
stage-1 训过的一切。`camera_token`/`register_token` 这 10240 个参数漏一个，物理-only
的目标就会悄悄把它们拖走，而 loss 曲线上完全看不出来（`5f1eff7` 修的就是这个）。

待决：

1. **关系怎么声明**：stage-2 的 config 里写一句 `continues_from: segvggt_finetune_agnostic`，
   由校验层去推导"必须冻的集合"？还是把关系写在 lock 文件里？
2. **约束是什么**：`frozen(stage2) ⊇ trained(stage1)`？还是更松的
   `trained(stage2) ∩ trained(stage1) = ∅`？两者对 joint 配方的含义不同——
   joint 同时训 `instance_` 和 `query_physgm`，它不是任何 stage 的续作。
3. **ckpt 侧要不要查**：`pretrained_weights` 加载的那个 ckpt 里，实际存在哪些键、
   哪些是随机 init（`query_physgm` by design 是 missing，`segvggt.py:297-320` 已有
   `_assert_query_physgm_coverage`）——这个已有的断言要不要并进指纹层。

**已有素材**：`src/model/arch/segvggt.py:297-320` 是这类断言的现成样板，可以直接抄形状。

## 完成判据

跨 stage 约束的形式定下来，且能对现有三份 segvggt 配方（stage-1 / joint / physgm）
各说清它该被约束成什么。

# 03 — 可训参数指纹里放什么

Type: grilling
Status: open
Blocked by: 01

## Question

**本图的核心决策。** 指纹要能同时抓住三类失败：漏冻、误冻、以及不走 `requires_grad`
的静默冻结（bf16）。

待决：

1. **粒度**：逐参数（几千行）/ 按顶层模块 rollup（十几行）/ 两级（rollup 给人看，
   逐参数哈希给机器比）？
2. **维度**：`requires_grad` 是必须的。**dtype 也是**（地图 Notes 第 2 条）。还要不要
   `param_groups` 归属 + 实际 lr？参数量？shape？
3. **形态**：人可读的清单（能进 code review、能 diff）还是一个哈希？还是两者都要——
   哈希做快速判等，清单做"差在哪"的定位？
4. **稳定性**：指纹必须对什么不敏感？参数遍历顺序、DDP rank、`num_semantic_classes`
   这类会改 shape 的 config 项——哪些进指纹哪些不进，决定它是好用还是天天误报。

**约束**：指纹必须在 `BaseWrapper.setup()` **之后、strategy wrap 之前**取得
（`base_wrapper.py:502-505` 的原因同样适用）。

## 完成判据

指纹的字段集合、粒度、序列化形态定下来，且能解释它为什么抓得住上述三类失败各一个
具体历史案例（`5f1eff7` 漏冻 token、`7e196e9` 误冻主体、`cbe93f9` bf16）。

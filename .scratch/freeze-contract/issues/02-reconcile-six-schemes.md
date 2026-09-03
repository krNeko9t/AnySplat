# 02 — 六种冻结实现：哪些留、哪些记录、哪些删

Type: grilling
Status: open
Blocked by: 01

## Question

`freeze_research.md` 分类出 6 种实现。给每一种定性，并划一条**"官方 vs 我们加的"**的线：

| # | 实现 | 位置 | 待定 |
|---|---|---|---|
| 1 | `freeze_keywords` + `BaseWrapper.setup()` | `base_wrapper.py:501-528` | 唯一入口，留 |
| 2 | `freeze_backbone` / `freeze_module` | `anysplat.py:157-193` | **是我们加的还是 AnySplat 官方的？** 无条件赋值分支（`:191-193`）语义与方案 1 相反 |
| 3 | LoRA 构造期冻基座 | `lora.py:123-125` | SegVGGT 官方，留 + 记录 |
| 4 | 教师网 `no_grad` + CPU 卸载 | — | 地图已判 out of scope |
| 5 | `mask_token.requires_grad_(False)` | `aggregator.py:302-304` | VGGT 官方，留 + 记录 |
| 6 | 推理期 `eval()` + 全冻 | — | 地图已判 out of scope |

真正要决的只有**方案 2**。已核实的事实（地图 Notes）：它只存在于 `anysplat.py`，
`segvggt.py`/`iggt.py` 零命中；用它的 5 份 config 无一设 `freeze_keywords`
⇒ **今天没有实际冲突，但没有任何机制保证明天也没有**。

选项：(a) 留着 + 让指纹层顺带把它的效果也算进去；(b) 留着但加一条启动期断言
"`freeze_keywords` 非空时 `freeze_module` 必须为 None"（把今天的隐式隔离变显式）；
(c) 判定为我们加的私货，删掉无条件赋值分支、把 5 份 config 迁到 `freeze_keywords`。

**先查 AnySplat 上游仓库确认方案 2 是不是官方代码**——这决定 (c) 是否违反地图
Notes 第 4 条。

## 完成判据

方案 2 的处置定下来；`freeze_research.md` 升级成契约文档的"现状"一节，每种实现标注
官方/自研 + 保留/删除。

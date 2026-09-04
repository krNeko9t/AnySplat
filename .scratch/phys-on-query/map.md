# Map: 让 SegVGGT 的 query 额外输出 object-centric P

Label: wayfinder:map

## Destination

跑出**第一个可交付的 checkpoint**：SegVGGT backbone + pretrained weight，query 在预测
class 之外额外输出 object-centric 的 P = (杨氏模量, 泊松比, 密度)，一次前馈、场景级、
推理不需要 GT mask。

**判据不是"物性预测得准"，是"学生逼近教师"**——即逼近当前这条 VLM 伪标签管线的上限。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）：导师要的是 checkpoint，
不是方案。票可以是动手的。

**领域**：前馈多视角 3D 场景理解 + 物性预测。ICLR 2027 截止 **2026-09-25**，
目标是**按时投出，不求录用**（memory `paper-goal-iclr2027`）。

**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

### 本图开工前已定的事（2026-09-03 grilling，不再重开）

1. **上限 = 教师**。前馈视觉底座在这个任务上的天花板就是生成伪标签的那个 VLM 管线。
   照片里没有杨氏模量；任何前馈模型在极限上都是查表，只是表的分辨率不同。
   **这条要写进论文第一段，自己说出来，不让审稿人来问。**
2. **上限由（输入数据分布 × VLM 本身 × 标签生成方法）三者决定**，未来一定会重跑标签来抬高它。
   **抬高上限 = out of scope；逼近上限 = 本图全部内容。**
3. **论文主张 = 系统 + 度量**：贡献是"把一条昂贵的 per-instance VLM 管线摊销成一次前馈，
   且 P 落在正确的实例上"，不是"模型理解物理"。同时诚实量出其中有多少是类别查表
   （已发表工作的主表里没人报过这个 trivial baseline）。
   "打不过查表"因此不是弱点，是自己报出的发现。
4. **物性头保持最朴素**：query → MLP → (mu, log var)，Gaussian NLL，不进 Hungarian 匹配代价。
   花招（如"类别项 + 残差项"分解）一律押后，等第一个 checkpoint 出来看它烂在哪再说。
5. **不以现有代码实现为准**。现有半成品只作为事实来源（`research_space/opus_2/notes/code_facts.md`
   带 file:line），不作为设计依据。

### 事实底座（可复核，别重新推导）

- `research_space/opus/01_审视结论.md` — 逐节点审视 + 可进论文的数字
- `research_space/opus_2/01_四问答复.md` — 四问答复 + 对上一轮的修正清单
- `research_space/opus_2/notes/code_facts.md` — 现场读码事实，带 file:line

关键数字：材质熵 H=3.865 bit，H(P|类别)=1.444 bit ⇒ **类别解释 62.6%，残差 37.4%**（Infinigen shader，
非 VLM 标签）；同场景同类别 ≥2 实例的实例占比 99.2%；29.6% 的物体跨材质原型。
SegVGGT Table 7：冻结 23.4 → LoRA joint **31.9**；Table 8：冻结底座下加大 head 22.4/23.4/**16.7**（倒退）。
⚠️ 以上是**论文数字，非本仓库实测**，不同数据/配方下不可直接套用。

## Decisions so far

<!-- 一行一个已关闭的票 -->

（暂无）

## Not yet specified

- **物性头的花招**：`P̂ = LUT[ĉ] + Δ(q)` 这类"类别项 + 残差项"显式分解。等第一个 checkpoint
  的失败模式出来再判断值不值。已知代价：依赖闭集 200 类的 `ĉ`，认错类时误差不再平滑。
- **学生超越教师的那条合法路径**：教师只看一个最佳视角，学生看全部视角 ⇒ 学生可以更一致、
  更抗噪。这是唯一不违反"上限=教师"的超越方式，可测，但度量怎么定还没想清楚。
- **part-level 粒度**：29.6% 的物体跨材质原型，object-level 单标签对它们是系统性错误。
  query 范式下拆 part 最便宜（多分配几个 query + 把 GT 拆到 part 粒度，不改表示）。
- **数据清洗**：墙面/地板/门窗/树木等不适合物理模拟的实例要不要剔除，剔除后训练分布怎么变。
- **训练/推理的 train-test 失配复核**：query 路径按 `code_facts` D 应当自动免除 GT-mask 依赖，
  但要在真实运行里确认一遍。

## Out of scope

- **抬高上限**：重跑伪标签（换视角选择 / 改 prompt / steering / agentic harness / 接外部材料数据库）。
  一定会做，但不在这张图里——这张图只做逼近当前上限。
- **视频监督反演物性**（看物体形变优化参数）。原理上是唯一真有视觉物理信号的路，我们做不到。
- **冻结机制本身**（字符串匹配脆弱、漏冻 token、多种实现并存）：
  另开一张图 `.scratch/freeze-contract/map.md`。本图只消费冻结配方，不改机制。
  **2026-09-04 更正**：原文把"bf16 静默冻结"也算进这条，是错的——它 `requires_grad` 全程为真、
  参数进了优化器，只占冻结三判据的第三条，**不是冻结**。freeze-contract 图已据此把它
  整票判出 scope（该图 07 号票），本图不能再把它推回去，否则两张图互指、它谁都不归。
  现寄存为本图 [06 号票](issues/06-aggregator-precision.md)（明确标注不在路上、不阻塞任何票）。
- **IGGT 路线**（`phys_iggt.yaml`）：aggregator/part_adaptor/part_head 全冻，phys head 是纯 frozen
  probe；要修得先给 VGGT aggregator 加 adapter，成本高一个量级。降级为"冻结底座"那一行的对照组。
- **复现 PIXIE / VoMP / PhysGS，MPM 灵敏度实验，HILO / PixieVerse**（memory `paper-direction-scene-phys`
  明令不碰）。

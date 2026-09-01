# R2 findings：跨视角 instance id 是否真的对齐

> 票：`.scratch/instancesplat/tickets/R2-跨视角id是否对齐.md`
> 方法：仓库内静态代码调查为主（未运行任何 python、未访问数据集）；上游数据生成过程无法从
> 本仓库判定，用 IGGT 论文原文佐证，**外部来源逐条标注**。
> 记号：**【事实】**= 代码/论文里直接读到的；**【推断】**= 我的分析。

---

## 0. 一句话结论

| 子集 | 跨视角 id 一致性 | 证据强度 | L_cross 是否可开 |
|---|---|---|---|
| `processed_scannetpp_v2` | **场景级全局一致** | **强**（3D 标注投影锚定 + 脚本按文件名配对 + 仓库已实训过） | **开** |
| `processed_re10k` | **构造上是全局的（tracker id），但残余漂移未量化**；**不是**「逐帧独立」 | **弱–中**（仅论文声明，无任何本地/上游量化） | **暂关**，待一次性验证脚本放行 |

票里「re10k 可能是逐帧独立 id」这个猜测**不成立**——它是 SAM2 视频跟踪产出的 track id，天生
跨帧。真正的风险不是「id 逐帧重排」，而是漂移导致的**同 id 指向不同物体**（见 §2.3）。

另有一条与本票同等重要的**代码事实**：`scripts/make_manifest_inscene_re10k.py` 写出的路径
缺少场景目录，re10k 现在**根本加载不了**（§1.3）。

---

## 1. Q1：两个 manifest 脚本怎么产生 `instance_mask`？

### 1.1 【事实】两个脚本都**不生成 id**，只是路径索引器

两份脚本从头到尾没有出现 `np.unique` / 重编号 / 偏移 / 任何对 id 值的写操作。它们只把
InsScene-15K 里**已存在的** `refined_ins_ids/*.npy` 的路径写进 manifest 的
`instance_mask_path` 字段。

- scannetpp_v2：`inst_dir = scene_dir / "refined_ins_ids"`
  （`scripts/make_manifest_inscene_scannetpp_v2.py:54`），
  写出字段 `"instance_mask_path"`（同文件 `:135`）。
- re10k：`inst_dir = scene_dir / "refined_ins_ids"`
  （`scripts/make_manifest_inscene_re10k.py:36`），写出字段（同文件 `:86`）。

**所以「id 是场景级全局还是逐帧独立」这个问题，这两个脚本给不出答案**——它 100% 继承自
上游 InsScene-15K 的 `refined_ins_ids`。两个子集用的是**同名目录**，且名字是 `refined_`
（= IGGT 匹配/合并**之后**的产物，不是裸的逐帧 SAM 输出）。答案要去 §2 找。

### 1.2 【事实】帧↔掩码的配对方式两个脚本不同，鲁棒性差一个量级

- scannetpp_v2 **按文件名 stem 配对**：先把 `refined_ins_ids/` 下的 `*.jpg.npy` 建成
  `mask_map[stem]`（`:83-88`），再用 metadata 里 `images[idx]` 的 stem 去查
  （`:95`, `:103`）；查不到就跳过该帧（`:104-105`）。**错配不可能发生**。
- re10k **按排序后的位置配对**：
  `zip(sorted(rgbs), sorted(insts), sorted(cams))`（`:46-48`, `:59`），前面只有一个
  数量相等的断言（`:52-56`）。

**【推断】** 位置配对是隐患：任一目录里混进一个多余文件（`.DS_Store`、临时文件），或三个
目录命名方案不同，整条序列就整体错位一格。错位的表现**恰恰就是「跨视角 id 完全对不上」**
——但那是我们的 bug，不是数据的问题。做 §5 的验证脚本时必须先排除这一项，否则会把自己的
bug 归因给 IGGT。

### 1.3 【事实】re10k 脚本的路径少了一层场景目录 —— 现在根本跑不了

```
# scripts/make_manifest_inscene_scannetpp_v2.py:132-135
scene_rel = scene_dir.relative_to(data_root)
"rgb_path": str((scene_rel / img_path.relative_to(scene_dir))),      # 带场景目录 ✔

# scripts/make_manifest_inscene_re10k.py:85-86
"rgb_path": str(rgb_path.relative_to(scene_dir)),                    # 不带场景目录 ✘
"instance_mask_path": str(inst_path.relative_to(scene_dir)),
```

而 `DatasetManifest` 的 `root` 是**整个数据集一个**（`src/dataset/dataset_manifest.py:96`），
解析方式是 `self.root / p`（同文件 `:134-136`）。于是 re10k 所有场景的第一帧都会解析到同一个
`root/rgb/xxxx.png`。合并 manifest 时 root 是 `InsScene-15K`
（`config/experiment/instseg_insscene15k.yaml:27`，配合
`scripts/merge_manifests_inscene15k.py:88-93`），路径必然不存在。

**旁证【事实】**：`git log` 里引入这两个脚本的提交 `b486ef9` 的信息原文是
「…ScanNet++ v2 manifest 设计与脚本实现；单独 ScanNet++ v2 数据集训练验证通过。**re10k未定**」。
即 re10k 这条路径从未被验证过，与代码读出来的结论一致。

---

## 2. Q2：上游标注管线（外部来源）

> 以下 §2.1–§2.2 全部来自 **IGGT 论文**（arXiv:2510.22706，`InsScene-15K Dataset` 一节）
> —— **外部来源**，不是本仓库代码。

### 2.1 【事实·外部】re10k（video captured）= SAM 起始帧 + SAM2 视频传播

- 第一帧用 SAM 出稠密 mask proposal，再把 proposal 当 prompt 交给 **SAM2 视频分割器沿时间
  传播**；
- 为处理新出现的物体、抑制漂移，采用**迭代关键帧**策略：未分割面积增大时重新在新关键帧上
  跑 SAM 去发现新物体；
- 整段视频处理完后，还有一次**双向传播（bi-directional propagation）**来提升 track 的
  时间一致性。

### 2.2 【事实·外部】scannetpp_v2（RGBD captured）= 3D 标注投影 + SAM2 细化 + 匹配合并

- 先把 **3D 标注投影到 2D** 得到带身份的初始 mask；
- 再用 SAM2 出**形状精细但没有身份**的 proposal；
- 最后把 proposal 与投影 GT **对齐以赋予一致的 object ID**，同 ID 的 proposal 合并成完整 mask。

论文并给出统一声明：**"we ensure that each instance retains a unique ID across all views."**
论文 A.7 承认的局限只涉及**边界精度**，未记录任何 ID 一致性的失败模式。

InstanceSplat 侧的对应表述（本仓库内）：ScanNet++ 用 IGGT 发布的 InsScene-15K 变体
"where instance masks are further refined for higher-quality supervision"，并"To align with
IGGT's training protocol"额外引入 RE10K 子集（5,137 scenes），用 T-mIoU / T-SR 评测跨视角
一致性（`ref_knowledge/InstanceSplat/InstanceSplat.md:192`）。

### 2.3 【推断】两者都「全局」，但保证的**性质不同**，这才是分歧点

- **scannetpp_v2 的 id 锚定在一个与视角无关的外部参照（3D 标注）上。** 任何两帧的 id 都是
  各自独立地从同一个 3D 物体导出的，不存在"沿时间累积"的误差通道。失效只能是**局部误配**
  （某帧某个 proposal 匹配错），不会系统性漂移。→ 全局一致，强。
- **re10k 的 id 是 tracker id，没有 3D 锚点。** 全局性来自"同一条 track 一路传下去"这个
  构造，因此有两类**不同后果**的失效：

  | 失效 | 现象 | 对 Eq.7 L_cross 的影响 |
  |---|---|---|
  | **碎裂 / 重编号**（重新关键帧后同一物体拿到新 id） | 同物体两个 id | 该物体退出 `K_i ∩ K_j` → **该项变成 no-op**，只是少了监督，**无害** |
  | **漂移 / id 碰撞**（某 id 的 mask 漂到了别的物体上） | 同 id 两个物体 | **主动把两个不同物体的原型拉到一起** ← 票里担心的、真正有害的那种 |

  论文提到的双向传播和关键帧重播种，主要针对的是**碎裂**；**漂移正是关键帧策略试图缓解的
  对象**，也就是说漂移被承认存在，但论文**没有给出残余漂移率**。
- **【推断】本项目特有的额外放大因素**：re10k 是连续视频轨迹，而我们的 view sampler 会在
  一个场景里挑 2–8 个 context view；一旦挑到时间上相隔很远的两帧，碎裂与漂移都处在最大值。
  且 re10k **没有深度**（`scripts/make_manifest_inscene_re10k.py:87` 注释；
  `src/dataset/dataset_manifest.py:209`、`:239-240`），几何上也没有任何别的约束来兜底。

**结论**：票里"re10k 可能逐帧独立"的猜测被证伪；但"re10k 的一致性明显弱于 scannetpp_v2"
这半句成立，且弱在一个**具体、可命名**的地方（漂移致 id 碰撞），而不是弱在"id 体系"。

---

## 3. Q3：`src/dataset/` 里有没有对 id 做重映射/压缩？

### 3.1 【事实】没有。一路读到 batch 张量，id 值原样保留

顺着 manifest → batch 的完整路径逐段核对：

| 环节 | 位置 | 对 id 做了什么 |
|---|---|---|
| 读盘 | `src/dataset/dataset_manifest.py:150-157` | `np.load(...).astype(np.int64)` → `torch.from_numpy`。**无 unique / 无重编号 / 无偏移** |
| 尺寸对齐 RGB | 同文件 `:224-229` | `F.interpolate(mode="nearest")`（先 `.float()` 后 `.long()`）。最近邻 → **id 值精确保留**，只改像素归属 |
| 堆叠成 `[V,H,W]` | 同文件 `:256-258` | 纯 `torch.stack` |
| 塞进 example | 同文件 `:276`（context）/ `:294`（target） | 直接放 `"instance_mask"` |
| crop / resize | `src/dataset/shims/crop_shim.py:107-118` | 调 `rescale_mask` |
| `rescale_mask` 本体 | `src/dataset/shims/crop_shim.py:37-57` | `cv2.resize(..., INTER_NEAREST)`；int64→int32 转换再转回，**值保留** |
| 数据增强 | `src/dataset/shims/augmentation_shim.py:27-28`、`:38-39` | 只做水平/垂直翻转，**不碰 id** |
| collate 成 `[B,V,H,W]` | `src/dataset/collate.py:19` | `default_collate` 纯堆叠 |

**结论【事实】**：票里担心的"最隐蔽的坑"——逐帧重映射把原本全局一致的 id 打散——**在本仓库
不存在**。`refined_ins_ids` 里有多少跨视角一致性，就有多少原封不动地进到 batch 张量。

**【推断】唯一的理论精度隐患**：`:224-229` 那次 `.float()` 中转在 id > 2²⁴ 时会丢精度。
InsScene 的实例 id 是小整数，实际不成立，但如果将来换数据源要记得。

### 3.2 【事实】id 只是**场景级**唯一，从不跨 batch 唯一 —— 给 T2 的硬约束

没有任何一处把不同场景的 id 错开。所以原型 loss **必须逐 batch item 分组**，跨 B 汇总会把
不同场景的同号 id 当成同一个物体。既有先例：`src/loss/loss_disc.py:243-252`，`multi_view`
分支写的是 `for b in range(B)`，只把**同一个 b 的各视角**合并算 per-instance mean。

### 3.3 【事实】既有 loss 已经依赖跨视角 id 对齐，但只在 infinigen 上开过

`loss_disc` 的 `multi_view: True` 会把同一 batch item 的所有视角摊平后算单一 per-instance
mean（`src/loss/loss_disc.py:243-252`），这**等价于隐式的跨视角一致性假设**。但：

- 默认关闭（`src/loss/loss_disc.py:54` `multi_view: bool = False`，`config/loss/disc.yaml`
  里根本没写这个字段）；
- 全仓库只有两个 experiment 打开它，且**都是 infinigen**
  （`config/experiment/instseg_inscene_infinigen_mv.yaml:49`、
  `config/experiment/instseg_iggt_infinigen_mv.yaml:43`）——infinigen 是合成渲染，
  逐物体 id 天然全局。

**【推断】** 即在本仓库历史里，`scannetpp_v2` 和 `re10k` 的跨视角 id 一致性**从来没有被任何
一次训练间接检验过**。这就是 scannetpp_v2 证据强度写"强"而不是"确证"的原因。

---

## 4. Q4：`instance_valid_mask` 覆盖多少像素？id 0 是背景还是未标注？

### 4.1 【事实】`instance_valid_mask` **在两个子集上都覆盖 100% 像素**——因为它根本不存在

- 数据集侧产出的是 `valid_mask`（深度有效性推导，
  `src/dataset/dataset_manifest.py:259-262`），键名就叫 `"valid_mask"`（`:275`、`:293`）。
- **三个 wrapper 都刻意拒绝把它转成 `instance_valid_mask`**，且都写了 I1 的理由注释
  （深度洞 ≠ 标注坏）：
  - `src/model/wrapper/iggt_wrapper.py:110`
  - `src/model/wrapper/anysplat_wrapper.py:178-179`
  - `src/model/wrapper/segvggt_wrapper.py:227-228`
- 各 loss 读的是 `depth_dict.get("instance_valid_mask")`，拿到的是 `None`
  （`src/loss/loss_disc.py:191`、`src/loss/loss_mvc.py:112`、
  `src/loss/loss_segvggt.py:286`）。

→ **今天的答案：两个子集都是 100%（全部像素有效）**。这不是数据属性，是接线状态。

**补充【事实】**：re10k 连深度都没有，`scene_has_depth` 为假时深度被填成全 1、`valid_mask`
被填成全 True（`src/dataset/dataset_manifest.py:239-240`、`:261-262`）。所以即便将来有人
把 `valid_mask` 接成 `instance_valid_mask`，在 re10k 上它也是**空转**的。

### 4.2 【事实】id 0 的语义：**未测定**，且只能从数据判定

`scripts/check_instance_id0.py` 的判据（**读脚本，未运行**）：

- 判别量不是 id-0 的**面积**而是它的**连通域形状**（`:10-18`）：真背景 = 一个贴边的巨大
  连通域；未标注物体 = 若干**内部**的紧致连通域。
- 实现：`ndimage.label` 后取不接触图像四边的连通域（`:82-89`），面积占比 `< 1e-4` 视为
  锯齿噪点丢弃（`:90`），累加得 `interior_frac`（`:94`）。
- 判决阈值（`:146-157`）：`interior_frac` 均值 `< 0.02` → 真背景；`0.02–0.08` → 边界情形；
  `> 0.08` → id 0 里藏着大量未标注物体。
- 脚本自己说明要**按源各跑一次**（`:24-25`）。

**【事实】仓库里没有任何一个子集的运行结果**（`docs/`、`.scratch/` 全域 grep 无 `interior_frac`
等输出痕迹）。所以这一问在本机无法回答，需要在集群上跑一次。

### 4.3 【推断】但 id 0 的语义**不阻塞 L_cross / T2**

Eq.4–7 全是对比式的：原型只在 `K_i`（该视角可见的**有效** id 集合）上定义，id 0 被整体
排除，不参与任何一项、也不产生梯度。仓库既有实现就是这么做的：
`src/loss/loss_disc.py:112`（`unique_ids = unique_ids[unique_ids != ignore_id]`，
`ignore_id` 默认 0，见 `:49` 与 `config/loss/disc.yaml:5`）、
`src/loss/loss_mvc.py:323`。T2 票也已经把 `instance_mask != ignore_id` 写进了 valid 判据。
这与 `docs/repo_knowledge.md:108` 的既有结论一致：**id 0 的含义对对比型 loss 无所谓，只对
集合预测型（SegVGGT）要紧**——而 SegVGGT 不在本 effort 范围内。

→ **不要让 §4.2 阻塞 T2。** 它是 SegVGGT 那条线的待办，顺手跑一下即可。

---

## 5. 【推断】要定论 re10k，需要什么样的一次性验证脚本

**先决条件**：必须先修好 §1.3 的路径 bug，否则脚本读不到文件；也必须先排除 §1.2 的错位
配对，否则会把自己的 bug 当成数据的 bug。

建议**一个脚本、三层判据**，全部是纯 numpy 读 `refined_ins_ids/*.npy` + 对应 rgb，**不需要
深度、不需要 GPU、不需要跑模型**：

1. **判"是不是全局"（决定性，无需几何）** —— 建 id×frame 占据矩阵，报告：
   每个 id 出现的帧数均值、出现 ≥2 帧的 id 占比、id 的出现帧是否为连续区段。
   逐帧独立编号会表现为"每帧 id 都是 1,2,3… 重复用、几乎没有跨帧 id"；tracker id 会表现为
   长连续区段。这一层就能把票里的原始猜测彻底证实或证伪。
2. **判"碎裂率"** —— 统计一条轨迹中途消失、随后同位置出现新 id 的次数（用质心 + 面积近邻
   匹配）。碎裂高只意味着 L_cross 信号变少，不致命。
3. **判"碰撞率"（真正要看的数）** —— 对每个在帧 i、j 都出现的 id，比较两帧中该 id 区域的
   廉价外观描述子（平均 RGB + 面积 + 质心位移）。碰撞（mask 漂到别的物体）通常表现为平均
   色的大跳变或质心的不连续跳跃。报告"描述子距离超阈值的 (id, i, j) 对占比"。

**放行建议**：在 re10k 上，若第 3 层的异常对占比足够低（数量级 ≤5%，具体阈值等真实分布出来
再定），就把 L_cross 的门打开；否则维持关闭。scannetpp_v2 上同一脚本跑一遍作为**对照基线**
——它的数值就是"3D 锚定应该长什么样"，re10k 与它的差距才是有意义的量。

顺手把 `scripts/check_instance_id0.py` 在两个子集上各跑一次，把 §4.2 一并结掉（服务
SegVGGT 那条线，不阻塞本 effort）。

**注意**：上面的"≤5%"是一次性分析口径，不是训练超参，不违反地图「不引入论文没有的超参」。
但 §6 引入的**门控开关本身是对论文的偏离**，必须在 T2 的 `## 解决` 里写明。

---

## 6. 建议：选 (a)，用 per-sample 门控来实现

**政策上选 (a)「只在 `scannetpp_v2` 上开 L_cross」；机制上用 (c) 的 per-sample 开关来落地。**
两者不矛盾——(c) 只是 (a) 的实现手段，因为一个 batch 里可能同时混着两个子集的样本。

不选 (b)「整个 re10k 不用」的理由：

1. **代价太大、收益太小。** Eq.5 pull 和 Eq.6 push 都是**视角内**的，完全不依赖跨视角 id
   对齐（`ref_knowledge/InstanceSplat/InstanceSplat.md` Eq.5/Eq.6，见票里引用）。re10k 依然
   能对 L_pull、L_push、L_rgb、L_bd-rgb、几何蒸馏贡献全量信号。只把 Eq.7 关掉，是**损失最小**
   的处置。
2. 丢掉 re10k 等于丢掉论文的 RE10K 部分（5,137 scenes）和它对应的 T-mIoU / T-SR 叙事
   （`InstanceSplat.md:192`），为一个**尚未量化**的疑虑付这个代价不划算。

不选"数据驱动的纯 (c)"的理由：那需要一个 per-pair 的 id 质量估计量，仓库里没有，且会引入
论文没有的超参 —— 地图明令禁止。

**门控怎么接（零侵入，不动 dataset 层）**：
`merge_manifests_inscene15k.py:33-38` 已经给三个子集打了 `inf_` / `spp_` / `re10k_` 前缀；
`dataset_manifest.py:282` 把它写进 `example["scene"]`；`src/dataset/types.py:33` 声明
`scene: list[str]`（长度 = batch size，`default_collate` 原样保留）；而每个 Loss 的
`forward` 都拿得到 `batch`（`src/loss/loss.py:29-36`）。所以新 loss **在 loss 层内部**读
`batch["scene"][b]` 判前缀即可，dataset 层一行不动。

---

## 7. 这个结论要求 T2 改什么

1. **Eq.7 变成有条件项。** `LossInsGroundCfg` 加一个门控字段（例如
   `cross_view_scene_prefixes: list[str] | None`，默认只放行 `spp_`）。这是**对论文的有意
   偏离**，按地图规矩必须写进 T2 的 `## 解决`：论文假设 GT id 已跨视角对齐，我们对 re10k
   无法确认这个前提。
2. **L_cross 必须逐 batch item 算，门控也逐 batch item 判。** 一个 batch 里 spp 与 re10k
   混排是常态，不能整批开或整批关。照抄 `src/loss/loss_disc.py:243-252` 的 `for b in range(B)`
   写法（这条同时也是 §3.2 的 id 只在场景内唯一所要求的）。
3. **新增一条边界情形**：整个 batch 全是 re10k 样本 → L_cross 恒为 0，但**必须留在 autograd
   图上**（与 T2 已列的"没有有效实例"同一个坑，`git log` 的 `fix(I3)`）。
4. **valid 判据要容忍 `instance_valid_mask is None`。** T2 现在写的是"`instance_mask !=
   ignore_id` **且** `instance_valid_mask` 为真 **且** 渲染 alpha 达标"。按 §4.1，
   `instance_valid_mask` 今天在两个子集上都是 `None`；如果实现成 `None → 全 False`，
   loss 会静默恒为 0。必须显式 `None → 全 True`。
5. **测试清单补两条**：
   - 双视角合成样例，**同一物体在两帧带不同 id** → cross 项应为 0（不是 NaN，也不能产生
     虚假拉近）；
   - 门控测试：`re10k_` 前缀样本 → cross = 0，`spp_` 前缀样本 → cross > 0，混合 batch
     两者并存。
6. **T2 不负责修 §1.3 的 re10k 路径 bug。** 那是数据侧的独立问题（建议单开一张票，归到
   T7 新训练配方或数据准备那条线），但**在它修好之前 re10k 一帧都进不了训练**，所以地图的
   数据配比在短期内事实上就是"只有 scannetpp_v2"。

**对地图数据配比的影响**：re10k 仍然参与训练（贡献 L_rgb / L_pull / L_push / L_bd / 蒸馏），
只是不进 Eq.7。若 §5 的验证脚本将来放行 re10k，**只需改一个配置项**即可翻转门控，不用改代码。

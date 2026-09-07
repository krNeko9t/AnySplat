# 仓库现状理解（供 AI 阅读，避免常见误判）

> 本文档描述仓库的**真实现状**（探索版，非定稿），供 AI 开发时对齐认知。
> 与 `memory.md` 配合阅读：memory.md 记录项目事实，本文档记录架构约定与"坑"。
> **新增能力的分层写法**以 `[layered_scheme.md](layered_scheme.md)` 为准（本文件偏现状与坑）。
> **术语**（冻结三判据、冻结入口、指纹、lock 等）以根 `[CONTEXT.md](../CONTEXT.md)` 为准，本文件不重复定义。

## 1. 这个仓库是什么

- 起点是 fork 的 **AnySplat**（3DGS 前馈重建），目标是在其上做**语义/实例分割推理**等新东西。
- 后来发现 AnySplat backbone 效果不佳，而 **IGGT**（输入输出相似的工作）更合适，于是把 IGGT 的模型代码**抠进本仓库复用**，共享已写好的数据集载入、训练框架、推理脚本。
- 所以本仓库是**多个仓库的缝合体**：AnySplat 框架（Hydra + Lightning + 数据管线）+ VGGT backbone + IGGT 的 instance head（SamProjector + PartHead）+ 自研的 PhysicsHead、trace 管线、坐标工具。
- **这是探索版本，不是定稿**。owner 要频繁尝试：不同 backbone × 不同 head × 不同 loss 的组合。**层与层的隔离是第一设计原则**——改一个小模块不应牵动其他层。
- **分层方案契约（新增能力默认遵守）**：见 `[layered_scheme.md](layered_scheme.md)`。Physics 已按该规范落地；instance 等旧路径尚未完全迁移。

关键事实（来自 memory.md）：

- 训练基于 AnySplat 预训练权重（重建部分已优化好），只有 instance head 等新增结构随机初始化。
- IGGT 官方**没有开源训练代码**，所以 loss（mvc / disc）是民间复现，可能有误。mvc 是像素级对比 loss，很难优化；disc 是实例级判别 loss，分单视角/多视角，加了 soft hinge，没按原文做 L2 归一化和 L2 正则。



## 2. 分层架构（最重要的一节）

从底到顶三层，**只允许上层依赖下层**；层间只通过 `src/model/outputs.py` 的命名契约传数据：

```
底层部件   src/model/vggt/  src/model/heads/  src/model/decoder/
           ├─ vggt/                  vendored VGGT backbone,整体一个盒子(models/ layers/ heads/ utils/)。
           │                         camera/dpt/track head 是预训练模型自带的,留在盒子里;
           │                         track_head 虽无人调用,但 VGGT.__init__ 无条件构造,
           │                         删了会破坏 from_pretrained 权重加载,必须保留。
           ├─ heads/                 自研 head,按领域分组:
           │   ├─ gaussian/          VGGT_DPT_GS_Head + GaussianAdapter(AnySplat 高斯路线)
           │   ├─ instance/          SamProjector、PartHead(IGGT 抠来)
           │   ├─ physics/           PhysicsHead/Classifier/pool + 三个 readout
           │   │                     + scheme.py(phys_scheme 装配:class/property/physgm_copy/physgm_dpt)
           │   └─ attention_blocks.py / window_attention.py  instance 与 physics 共享的注意力层
           └─ decoder/               splatting CUDA 渲染 decoder(目前只有 AnySplat 用)

契约       src/model/outputs.py  EncoderOutput + PhysicsPrediction / PhysicsPropertyPrediction /
           src/model/types.py    PhysGMPrediction;Gaussians。encoder 产出、wrapper/loss 只读。

arch 层    src/model/arch/     模型组装,一条路线一个文件
           ├─ base.py               Encoder 抽象基类
           ├─ anysplat.py           EncoderAnySplat(VGGT+GS head,可选 instance head)
           │                        + AnySplat(encoder+decoder 壳,HF mixin)
           ├─ iggt.py               EncoderIGGT(VGGT+PartHead,可选 physics scheme,无 GS head)
           │                        + IGGTModel(纯 encoder 壳 + 官方 ckpt 键名重映射加载)
           ├─ weight_loading.py     HF/Lightning/run_dir 权重加载权威实现
           └─ __init__.py           get_model(encoder_cfg, decoder_cfg):按 cfg 类型分发;
                                    EncoderCfg union 唯一权威

wrapper 层 src/model/wrapper/   负责"再包一层训练",对接 Lightning 训练/推理
           ├─ base_wrapper.py       BaseModelWrapper:optimizer 分组、loss 汇总、日志、可视化等公共逻辑
           ├─ anysplat_wrapper.py   渲染重建训练(decoder 渲染 → mse/lpips/depth 等)
           └─ iggt_wrapper.py       encoder-only 训练(instance/physics 特征 → disc/mvc/phys loss),无渲染

入口       src/main.py          Hydra 入口;scripts/ 下各推理/导出脚本
```

Import 约定:跨目录一律 `from src.model.xxx import ...` 绝对导入,同目录兄弟可用单点相对;vggt/ 内部维持 vendored 原样。Hydra config group 仍叫 `config/model/encoder/`(yaml 组名与代码目录解耦,未随代码搬家)。

**统一的层间契约**：所有 encoder 的 forward 返回 `EncoderOutput`（`src/model/outputs.py`），字段包括 `gaussians`（IGGT 为 None）、`pred_context_pose`、`depth_dict`、`instance_feat_map [B,V,N,H,W]`、`gaussian_instance_feat`、`physics_prediction`（`PhysicsPrediction | None`）。**新增 head 输出时，往 EncoderOutput 加可选字段（默认 None），不要改已有字段语义**——这是 wrapper 与 loss 之间解耦的接口。

**分发点（改组合时要看的三个注册表）**：

1. `src/model/arch/__init__.py` 的 `MODELS` / `get_model`：按 `encoder_cfg` 的 dataclass 类型分发到 arch。
2. `src/main.py` L128：按 `isinstance(cfg.model.encoder, EncoderIGGTCfg)` 选 `IGGTWrapper` 还是 `AnySplatWrapper`。
3. `src/loss/__init__.py` 的 `LOSSES` 字典 + `src/dataset/__init__.py` 的 `DATASETS` 字典。

新增 backbone/arch/head/loss 时，照抄现有 iggt 的接线方式：新 cfg dataclass → 注册到对应字典/Union 类型 → 加 `config/model/encoder/` 或 `config/loss/` 下的 yaml → 用 `config/experiment/` 组合。

### 算法登记表

一个「算法」的身份 = **(parser, heads, losses)** 三元组，权威定义就是对应的 experiment yaml（本表只是索引，改组合以 yaml 为准），此设计为暂时弱约束，以后按需求落地为**algorithm 层**。仓库现有 4 套算法配置：


| #   | 算法                      | experiment yaml                                                                                    | encoder(arch) + heads                                                                                           | dataset                  | parser                       | losses                   |
| --- | ----------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------- | ------------------------ |
| 1   | 前馈 3DGS 重建（AnySplat 原版） | `multi-dataset.yaml`                                                                               | anysplat + splatting decoder；GS head（DPT）                                                                       | dl3dv + co3d + scannetpp | —                            | mse, lpips, depth_consis |
| 2   | 前馈 3DGS + 实例            | `instseg_anysplat.yaml`（变体：`instseg_small` / `instseg_inscene_infinigen*` / `instseg_insscene15k`） | anysplat（`instance_feat_dim=8`）+ decoder；GS head + SamProjector + PartHead                                      | manifest                 | —                            | mse, lpips, disc         |
| 3   | 实例（IGGT 路线）             | `instseg_iggt.yaml`（变体：`instseg_iggt_infinigen_mv`）                                                | iggt（encoder-only）；SamProjector + PartHead                                                                      | manifest                 | —                            | disc                     |
| 4   | 实例 + 物理分类               | `phys_iggt.yaml`                                                                                   | iggt（`phys_scheme: class`）；SamProjector + PartHead（冻结）+ PhysicsHead + PhysicsClassifier | manifest                 | `physics_parser: 3dovs_json` | phys                     |
| 5   | 实例 + 物理量回归             | `phys_prop_iggt.yaml`                                                                              | iggt（`phys_scheme: property`）；SamProjector + PartHead（冻结）+ PhysicsHead + PhysicsPropertyReadout | manifest                 | `physics_parser: instascene_vlm` | phys_prop                |
| 6   | SegVGGT 端到端实例（iter 1 推理 / iter 2 训练 / iter 3 物理挂 query） | `segvggt_finetune_agnostic.yaml`（stage-1 class-agnostic 微调）；`segvggt_scannet.yaml`（论文全量训练）；`segvggt_physgm.yaml`（stage-2 仅训物理 readout）；`segvggt_agnostic_phys_joint.yaml`（mask+objectness+phys **联合训**）；`scripts/segvggt_infer.py`（推理，含物理导出）                                             | segvggt（encoder-only，object queries）；vendored box `src/model/segvggt/` + SemanticHead（+ 可选 `QueryPhysGMReadout`）                          | manifest（instance_mask+相机+深度；物理时 `physics_parser: instascene_vlm_physgm`）  | —（class-agnostic 默认，可选 semantic）；物理 = `PhysGMTarget` | segvggt（cls+BCE+Dice+FADA+物理）, segvggt_geo（camera+depth） |


新增第 5 套算法时：新建 experiment yaml 组合三元组，并在本表加一行。

### SegVGGT 路线（新增，iteration 1 = 仅推理）

与 IGGT 不同：SegVGGT 把实例推理做进 transformer 内部——object queries 在每层 global attention 后 cross-attend 图像 token，每个 query 直接出「per-view mask + 类别分布（末通道 = no-match，未匹配空槽；DETR 文献常称 no-object）」，端到端、无聚类后处理、无 GT-mask 池化。这正是把 per-object 物理属性挂在 query 上的天然载体（iter 3 目标）。

- **vendored box**：`src/model/segvggt/`（改版 aggregator + CrossBlock/CrossAttention + SemanticHead + LoRA；类别约定已矫正，见下）。与现有 `src/model/vggt/` box 隔离，互不牵动。不搬 `dependency/`（VGGSfM tracker）与 `utils/geometry`。
- **类别约定（唯一收束点）**：`src/model/segvggt/utils/scannet_instance_taxonomy.py`。配置写 `num_semantic_classes: 20|200` + `non_instance_classes: [wall, floor]`；头宽 = `(num_semantic - len(non_instance)) + 1`（no-match）。wall/floor 是 ScanNet 语义 stuff、实例无标、benchmark 不评——用显式列表表达，不是隐含 −2。叙述见仓库根 `docs/segvggt_scannet20_推理与类别数.md`。
- **arch**：`src/model/arch/segvggt.py`（`EncoderSegVGGTCfg` / `EncoderSegVGGT` 持有 vendored `SegVGGT` 为 `self.model`，forward 重打包成 `EncoderOutput`；`SegVGGTModel.from_checkpoint` 给官方 `.pt` 键加 `encoder.model.` 前缀后 shape 对齐 strict=False）。已注册进 `MODELS` / `EncoderCfg` union / `get_model`。
- **契约槽**：`EncoderOutput.segvggt_prediction`（`SegVGGTPrediction`：query_masks / query_class_logits / query_embed / feature_map / attn_frame_mean / query_phys_mu / query_phys_var / property_names）。`query_embed` 已在 iter 3 接出（见下）。
- **config**：`config/model/encoder/segvggt.yaml`（`num_semantic_classes` + `non_instance_classes`；LoRA rank 32 必须与训练一致）。
- **权重**：官方 HuggingFace `JinyuanQu/SegVGGT`（`checkpoint/segvggt_scannet{v2,200}.pt`，各约 6.6GB，含 DINO backbone，非 `hf:` 前缀走 from_checkpoint）。**官方无训练代码**，loss（Hungarian + BCE/Dice + FADA JS + teacher 蒸馏）需 iter 2 民间复现。
- **验证到位（iter 1）**：本仓库构建的 state_dict 键集与官方 `SegVGGT`（同 eval 配置）**逐键一致（2606=2606，零差异）**→ 官方 ckpt 加载零 bad-missing/unexpected；随机权重 CPU 端到端小前向 shape 全部打通。真权重前向/掩码质量待集群跑（本机无 GPU、缺 gsplat/hydra，仅开发机）。

#### iteration 2 = 训练复现（官方无训练代码，对照论文民间复现）

论文（`~/tmp/move_segvggt/segvggt/SegVGGT.md` + `mental-model.md`）给了完整训练配方：`L_total = L_geo + L_inst + λ_js·L_js`。官方只发权重不发训练码，故按 IGGT 的 disc/mvc 先例民间复现。**全部落在 loss 层（纯公式，无可学习参数）+ wrapper 层**，backbone box 与 arch 契约零改动。

- **matcher**：`src/loss/segvggt_matcher.py`（纯 torch+scipy，无仓库依赖，可单测）。DETR/Mask2Former 式二部图匹配，cost = `-λ_cls·c_{j,ck} + λ_mask·(BCE+Dice) + λ_js·C_js`（论文 Eq.5）。多视角 mask 摊平成 `N·H·W` 即等价点云 mask，直接继承点云分割的 BCE+Dice。`C_js`=帧级注意力 JS 散度（Eq.8）。
- **实例 loss**：`src/loss/loss_segvggt.py`（`LossSegVGGT`，注册名 `segvggt`）。跑 matcher → `λ_cls·CE(带 no-match 通道) + λ_mask·(BCE+Dice) + λ_js·L_js(FADA)`。GT 全部从多视角 `instance_mask` 现推：每实例二值 mask（area 降采到预测分辨率）、帧可见度分布（面积占比，用于 FADA）、类别（默认 class-agnostic；给 `instance_semantic` 逐像素图且 `class_agnostic=false` 则多数投票转类别监督）。
- **id 0 与欠标注对本 loss 的影响（和 IGGT 不同，但没有此前说的那么严重）**：IGGT 的 `loss_disc`/`loss_mvc` 是对比型，只拉近/推远同异 id 像素，对 id 0 **不产生任何梯度**，所以 id 0 是"背景"还是"未标注"无所谓。SegVGGT 是集合预测，但要分清两侧：**mask 侧不说谎**——匹配到实例 k 的 query，其 mask 目标是 `(inst_mask==id_k)`，id-0 像素目标为 0，含义是"不属于实例 k"，这句话即便 id 0 里藏着未标注物体也仍是真的。真正的问题在**分类侧**:若某 query 恰好盯上那个未标注物体，它不会被匹配，于是被判 `no-match`——这才是被"教错"的地方，而 DETR 系方法普遍不专门处理它。因此:(1) 配置开关 `loss.segvggt.unlabeled_as_ignore`（默认 false）把 id-0 从 mask 里摘掉,**瞄错了地方**——mask 侧本无谎,摘掉反而删真实负样本、致 mask 膨胀,保持关闭;(2) 但基于 `instance_valid_mask` 的 ignore **仍正确**,理由不同:那些像素 **id 本身不可信**,"这像素不是实例 k"可能是假话。实现:matcher 和 loss 的 BCE/Dice 都接逐像素 `keep` 权重(两侧须一致),被忽略像素既非正也非负。`scripts/check_instance_id0.py` 仍可用于摸清三个源 id-0 语义(按连通域形状:真背景=贴边大连通域,未标注物体=内部紧致连通域),但**不是 iteration 1 阻塞项**。
- **class-agnostic 的正确做法 = 边缘化，不是"打成 class 0"**：本仓库 manifest 数据集只有多视角一致 instance id，没有语义标签。若把 GT 全打成 class 0，等于逼预训练头把 channel 0（ScanNet 的 "cabinet"）改造成"任意物体"、同时压掉另外 17 个通道，预训练的 objectness 先验被丢光。正解是对类别分布做边缘化：`P(object) = Σ_c P(c) = 1 - P(no-match)`，实现为在 `[logsumexp(前景 logits), no-match logit]` 上做二分类 CE（matcher 的 `cost_cls` 同步变成 `-P(object)`）。好处：预训练分类头**一个权重都不用改**（无 head surgery、ckpt 键不变）、与推理侧 `1-P(no-match)` 判据（`scripts/segvggt_infer.py`、wrapper 的 `val/queries_fired`）自洽、18 个前景通道的语义结构保留下来供 iter 3 的 per-query 物理读出使用。**注意：这不代表 init 时权重就已 class-agnostic**——ScanNet 把未标注物体当背景训过,init 时 `1-P(no-match)` 的含义是"是不是那 18 类之一",不是"是不是任意物体"。边缘化只是给微调一个合理起点,把"任意物体"教进去仍要靠微调;**起始 loss 不必然低,不能当验收标准**。
- **FADA 关键接线**：vendored forward 已出 `attn_frame_mean`（`[L,B,Q,S]`，= 每帧注意力质量的 `1/P·Σ`，即论文 `p̂` 差一个逐帧常数 P）；loss 里**沿帧维重归一化**即恢复论文归一化分布。FADA 双角色：既是 matcher 的 cost 项（`cost_js>0`），又是 matched pair 的正则 loss（`λ_js`）——两者共享一次匹配，故合在同一个 Loss 内算。
- **几何 loss**：`src/loss/loss_segvggt_geo.py`（`LossSegVGGTGeo`，注册名 `segvggt_geo`）。`λ_camera·L_camera + λ_depth·L_depth`。camera=9 维 pose encoding 上的 Huber（沿 camera head 迭代 γ 衰减，复用 `loss_huber`）；depth=按序列单一 median scale 对齐后 L1+梯度（VGGT 上到尺度）。监督源解耦：wrapper 经 `depth_dict["segvggt_geo_target"]` 喂 target——默认 `gt`（manifest 干净相机/深度，恒可跑），可选 `teacher`（冻结 VGGT 蒸馏，论文原味，需 1B 权重+全环境，`train.segvggt_geo_supervision: teacher` 开）。
- **wrapper**：`src/model/wrapper/segvggt_wrapper.py`（`SegVGGTWrapper`，镜像 `IGGTWrapper`，encoder-only 无渲染）。全部监督帧在 `context`（`num_target_views: 0`，无 NVS target）；把 `segvggt_prediction`/`instance_mask`/`pred_pose_enc_list`/`depth`/`geo_target` 塞进 `depth_dict` 交给 loss。`src/main.py` 分发新增 `isinstance(EncoderSegVGGTCfg)→SegVGGTWrapper`（在 IGGT 分支之前）。
- **超参对齐论文 A.4**：`λ_camera=5, λ_depth=1, λ_cls=0.5, λ_mask=1, λ_js=0.5`，matching cost 权重=loss 权重；新参数 lr `2e-4`、pre-existing `6e-5`（=6e-5×3.333 param_group）、DINO(`patch_embed`) 冻结、LoRA rank 32、梯度裁剪。experiment：`config/experiment/segvggt_scannet.yaml`。
- **验证到位（iter 2）**：CPU 合成张量端到端跑通 matcher（二部图分配合法、cost 有限、JS(p,p)=0）+ `LossSegVGGT`（cls/BCE/Dice/FADA 四项有限、matched 计数正确、梯度回传到 query mask/类别 logits、空场景→仅 no-match CE、`cost_js=0` 退化为 loss-only FADA、class-agnostic 边缘化后梯度覆盖全部 18 个前景通道且 `P(obj)=1-P(no-match)` 恒等式精确成立、class-aware 回退路径可跑）+ `LossSegVGGTGeo`（camera/depth 有限、梯度回传、缺 target→0）。真数据训练/收敛待集群（本机无 GPU）。

**stage-1 class-agnostic 微调**（`config/experiment/segvggt_finetune_agnostic.yaml`）——拿官方 ckpt 直接适配成 class-agnostic，用来体检训练实现：

- 冻结集 `[patch_embed, frame_blocks, global_blocks, camera_head, depth_head, camera_token, register_token]`，即整个 aggregator（含 ckpt 里已训好的 LoRA）+ 几何头 + bare tokens 全冻；**只训** instance 分支 + 实例分类头（`semantic_head`）= **487M 可训**（AdamW 动量约 3.9GB）。几何不可能漂移，故 `segvggt_geo.weight: 0`。
- 因为没有任何随机初始化的新参数（是适配不是从头训），论文的 `2e-4` 太烫，改单一 `lr: 2e-5`。
- 对照：论文全量配方 `segvggt_scannet.yaml` 冻结集只有 `[patch_embed]`，可训 **1148M**（AdamW ~9.2GB）。注意 `[patch_embed]` 并不等于"冻住 aggregator"——LoRA 的覆盖面见 §6.2。
- 实测参数量核对（`paper_repo` env CPU 建模）：`patch_embed`→344 个/304M（DINO）、`instance_`→1011 个/454M、`semantic_head`→62 个/32.6M、`lora_`→192 个/9.44M（96 层 × qkv+proj）。yaml 里的 keyword 全部命中，无空组。

**joint = class-agnostic 分割 + per-query 物理同训**（`config/experiment/segvggt_agnostic_phys_joint.yaml`）——Phase-1 主路径，替代「stage-1 再 stage-2」的串行冻结：

- 同一 object query 上同时开 `λ_cls/mask/js/phys`（共用 Hungarian）；`phys_scheme: query_physgm`；分类侧仍边缘化 18+1（暂不做 2 路头手术）。
- 冻结集同纠正后的 stage-1（aggregator + geo + camera/register tokens）；可训 = `instance_` + `semantic_head` + `query_physgm`。随机 init 的 readout 用 `param_groups` 提到 ~1e-4。
- 数据 = phys scannet100（`instascene_vlm_physgm`）；无物理标注的样本上 `lambda_phys` 静默为 0。
- 与 `segvggt_physgm` 的区别：后者冻死分割只训 ~0.2M readout；joint 让 mask/objectness/phys 一起塑造 query。

#### 训练期可观测性（分割）

训练只有 loss 曲线时，"loss 在 0.6 震荡"无法区分**还没收敛**和**目标压根没在被优化**。`src/evaluation/instance_metrics.py` 补上实例指标（`src/evaluation/metrics.py` 只有 PSNR/位姿/深度，没有实例这一类）：

- **诊断量**：`matched_iou_mean`（IoU 最优 1-1 指派下的平均 IoU——正是 mask loss 在最大化的量，应从 ~0 爬到 >0.5，**最该盯的一个标量**）、`best_iou_per_gt_mean`（不做指派，每个 GT 取全体 query 最大 IoU；与前者的差距能区分"掩码有了但指派/分类不对"和"根本没学到"）、`n_gt`/`n_fired`/`fired_over_gt`（抓两种塌缩：全判 no-match、或全部 query 激活）、`mask_area_mean`（掩码膨胀）。
- **基准量**：`ap25`/`ap50`/`ap`（ScanNet 惯例，按 score 排序贪心配对 + all-point 插值）。
- **口径与训练完全一致**（否则数字不可比）：IoU 在摊平的 `S*h*w` 体上算；GT 用与 `LossSegVGGT._downsample_masks` 相同的 area+0.5 阈值降采样；`instance_valid_mask` 像素按 loss 的 `keep` 同样剔除；objectness 判据 `1-P(no-match)` 与推理侧一致。
- 接线在 `SegVGGTWrapper.validation_step`：全部指标 `self.log("val/...")`，并把对比图从 `Context|Depth` 扩成 `Context|GT inst|Pred inst|Depth`。**注意预测图与 GT 图调色板互相独立**（query 下标与 instance id 无对应关系），看形状不要看颜色。
- `SegVGGTModel.from_checkpoint` 现在也吃微调产出的 Lightning `.ckpt`（剥 `model.` 前缀，用 `model.encoder.` 存在性做门卫，官方 `.pt` 不受影响）——此前没有任何办法拿微调权重跑推理。

#### iteration 3 = per-object 物理挂在 object query 上

**关键前提：物理那套基础设施本仓库早就有**（`src/dataset/physics/` 解析器与三种 Target、`src/model/heads/physics/`、`loss_phys*.py`、5 个 IGGT experiment），属性固定为 `density / youngs_modulus / poisson_ratio`。iter 3 不是从零写，而是把 object query 接上去。

- **与 IGGT 物理路线的本质区别**：`PhysGMReadout` 必须用 **GT instance mask** 做 masked average pooling 才能造出 per-object token，因此推理时没有 GT 就跑不了。SegVGGT 的 query 本身就是 per-object 向量，`QueryPhysGMReadout`（`src/model/heads/physics/query_physgm_readout.py`）直接在 `[B,Q,D]` 上解码出 `(mu, var)`——**无 mask、无池化、无聚类，推理时对未标注新场景可用**。解码器从 `physgm_readout._make_property_decoder` 复用，两条路线不会漂移。
- **暴露 query**：vendored `src/model/segvggt/models/segvggt.py` 加一行 `predictions["instance_queries"] = instance_queries`（**加性改动**，只多一个 dict key）。给出的是**投影前的 1024 维**向量，不是 128 维的 `instance_queries_for_mask`——后者是为 mask 点积训出来的瓶颈，读出头自带 LayerNorm+Linear，喂全向量信息更全。
- **loss 折进 `LossSegVGGT`（`lambda_phys`，默认 0）而不是新建 loss 文件**：物理项必须用**与掩码完全同一次匈牙利匹配**。独立 loss 只有两条路——重算匹配（两边 cost 权重一旦不一致就静默分配到不同 query），或依赖 loss 之间的执行顺序（本仓库任何地方都没有这种耦合）。而"共享匹配所以折在一起"在该文件里**已有先例**：FADA 就是这么处理的。
- **`_build_targets` 现在返回第 4 项 `ids`**。`g_idx` 是**行号**不是 instance id，而 `PhysGMTarget.value_lut` 按**真实 instance id** 查表，必须走 `ids[g_idx]`。这是本迭代最容易写错的一处，已用非连续 id（3/11/40）+ 逐属性 MSE==0 的判定性测试锁死。
- **stage-2 配方**（`config/experiment/segvggt_physgm.yaml`）：`pretrained_weights` 吃 stage-1 的 `.ckpt`；冻结集加 `semantic_head` 与 `instance_`（一个关键词覆盖 `instance_query_token`/`instance_cross_blocks`/`instance_queries_proj`/`instance_query_self_attn`），只训新头；`lambda_cls/mask/js` 归 0（对应模块已冻，算了梯度也无处可去），但**匹配仍然要算**——`lambda_phys` 依赖它。数据 = 与 `physgm_iggt.yaml` 同一批 scannet100（那批才有 InstaScene VLM 物理标注）。
- **推理产出**：`scripts/segvggt_infer.py` 的 `decode_instances` 多返回 `query_idx`，据此把 `query_phys_mu/var` 对齐到存活实例，用数据集自己的 `physgm_denormalize` 反归一化成 SI，打表并写 `physics.npz`。这是整条路线的终点产物。
- **验证到位（iter 3，CPU 合成张量）**：读出头形状/`var>0`/梯度；`lambda_phys>0` 时 loss 有限且梯度回到 mu/var；`ids[g_idx]` 判定性测试（非连续 id 下逐属性 MSE==0，错位对照 MSE>1）；`PhysGMTarget.valid` 正确剔除未标注实例；缺 target / 缺读出头时该项静默休眠不崩；**回归护栏：`lambda_phys=0` 时与改动前的 loss 逐位相同（diff=0.00e+00）**；三个 segvggt experiment 配置的 encoder/loss dacite union 全部 MATCH OK；推理侧 SI 反归一化与 `physgm_denormalize` 一致。真数据训练/收敛待集群。

## 3. 权重加载（按来源分流，权威在 arch）

实现集中在 `[src/model/arch/weight_loading.py](../src/model/arch/weight_loading.py)` + `[get_model](../src/model/arch/__init__.py)` + `[IGGTModel.from_checkpoint](../src/model/arch/iggt.py)`。**scripts 禁止再复制** `_load_lightning_ckpt` / HF `strict=False` 白名单；新脚本只调这些入口。

四类旋钮（不要混成一个假统一 API）：


| 阶段                          | 旋钮                           | 权威实现                                                                              |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| 建模时灌预训练                     | `encoder.pretrained_weights` | `get_model` → HF 走 `init_anysplat_from_hf`；IGGT 本地路径走 `IGGTModel.from_checkpoint` |
| 训练 resume（含 optimizer/step） | `checkpointing.load`         | Lightning `Trainer.fit(ckpt_path=...)`（`src/main.py`）                             |
| 推理加载训后权重                    | `--ckpt` + `run_dir`         | `load_model_from_run`                                                             |
| 快速 demo（无 run_dir）          | 脚本 `--hf_model`              | `init_anysplat_from_hf`（与 `get_model` 共用白名单）                                      |


「手里有什么 → 用哪个」：


| 手里有什么                          | 用哪个旋钮                                                      |
| ------------------------------ | ---------------------------------------------------------- |
| HF 发布的 AnySplat / 配置里 `hf:...` | `encoder.pretrained_weights`（训练）或 `--hf_model`（无 run 的脚本）  |
| IGGT 官方/本地 `.pth`（需键名 remap）   | `encoder.pretrained_weights`（非 `hf:` 前缀）                   |
| 自己训出的 Lightning `.ckpt`        | 训练 resume → `checkpointing.load`；推理 → `--ckpt` + `run_dir` |


细节：

- **AnySplat HF**：`init_anysplat_from_hf` 加载后 `strict=False` 灌入；新增 head 的 missing keys 靠 `ALLOWED_ANYSPLAT_MISSING_PREFIXES`（`encoder.instance_head.` / `encoder.part_adaptor.` / `encoder.part_head.` 等）放行。**新增 head 后必须把前缀加进该常量**，不要在脚本里另写一份。计数与 bad_missing 前缀同样会 print（见上）。
- **IGGT 官方 ckpt**：`IGGTModel.from_checkpoint` 做两步键名重映射（`part_head.scratch.X → part_head.X`；补 `encoder.` 前缀），再按 shape 对齐后 `strict=False`。
- **VGGT backbone**：encoder 构造时 `VGGT.from_pretrained("facebook/VGGT-1B")`，是构造副作用，不是用户旋钮。
- **Lightning** `.ckpt`：`load_lightning_state_dict` 剥 `state_dict`；`load_model_from_run` 读 `run_dir/.hydra/config.yaml` → `get_model` → Wrapper → `load_state_dict`，返回 `wrapper.model`（只要 encoder 则取 `.encoder`）。`strict=False` 的 missing/unexpected 计数会 **print 到 stdout**（不只靠 logger），因为推理脚本通常未配 logging。



## 4. loss 体系

- 基类 `src/loss/loss.py`：`forward(prediction: DecoderOutput, batch, gaussians, depth_dict, global_step)`。
- **约定（有点脏但是现状）**：IGGT 路线没有 decoder 输出，`IGGTWrapper.training_step` 把监督信号塞进 `depth_dict` 传给 loss（`prediction`/`gaussians` 传 None）。instance 路线仍用裸 key（`instance_feat_map` / `instance_mask`）；**physics 路线用结构化对象**：`physics_prediction`（`PhysicsPrediction`）+ `physics_target`（`list[PhysicsTarget]`）。
- 分割相关 loss：`loss_disc.py`（实例判别，主力）、`loss_mvc.py`(像素对比，难优化)、`loss_phys.py`（物理属性分类——**纯公式，无可学习参数**；classifier 在 encoder 的 `PhysicsClassifier`）。
- loss 的开关和权重完全由 Hydra 的 `loss: [disc]` 列表 + `config/loss/*.yaml` 控制，代码里没有 if 开关。

Physics / 通用分层契约、改需求指哪里、反模式：见 `[layered_scheme.md](layered_scheme.md)`（权威）；Cursor rule：`.cursor/rules/layered-scheme.mdc`。

## 5. 数据管线

- 主力数据集是 `src/dataset/dataset_manifest.py`（`name: manifest`）：**manifest.jsonl 驱动**的通用多视角数据集，InsScene-15K 各子集（infinigen / scannetpp / re10k）都走它。manifest 由 `scripts/make_manifest_*.py`、`scripts/extract_and_make_manifest_insscene.py`、`scripts/prepare_3dovs.py` 生成。
- **manifest schema 是唯一方言**：frame 字段固定为 `rgb_path` / `instance_mask_path` / `K_px` / `c2w`（必填）+ `depth_path` / `HW`（可选），scene 字段为 `scene_id` + `frames`。加载端**不接受别名字段**；新生成脚本必须产出此格式，不要往加载端加兼容分支。
- Physics 监督：`dataset.manifest.physics_parser` 指向 `src/dataset/physics/parsers.py` 注册表；产出顶层 `batch["physics_target"]`（`list[PhysicsTarget]`，collate 在 `src/dataset/collate.py`）。dataset 按**监督通道**扩展（可选字段 + parser 插槽），不按算法拆类。
- **坐标约定（写死的，manifest 必须遵守）**：c2w 为 **OpenCV 相机到世界** 4x4；K 为像素单位 OpenCV 约定（COLMAP 内参需先减 0.5，用 `src.coord.colmap_to_opencv_intrinsics`）。加载时**不做任何坐标转换**。所有坐标转换统一走 `src/coord/`（CameraPose / CameraConvention / ExtrinsicType），**不要在脚本里手写 blender2opencv 矩阵**（历史上就是这么出的错，git log 里有清理记录）。
- 图像张量约定：dataset 输出 `[0,1]`，normalize shim 把 **context** 收到 `[-1,1]`，encoder 吃 `[0,1]`——wrapper 里有 `(image + 1) / 2`。NVS 的 target 保持 `[0,1]` 作渲染 GT，不要 shim。
- SegVGGT / IGGT：**无 NVS target**。实验钉 `view_sampler.num_target_views: 0`；manifest 空 target 时省略 `target` 键；wrapper 若收到非空 target 会 raise。全部监督帧只在 `context`，train/val 同契约。



## 6. 参数冻结与 freeze lock 契约

术语（冻结三判据 / 冻结入口 / 构造期冻结不变量 / 指纹 / lock / bf16 死参数）在根
[`CONTEXT.md`](../CONTEXT.md)，本节只写机制现状。

### 6.1 唯一入口

**config 能拨动的冻结开关只有 `optimizer.freeze_keywords`**（`src/model/wrapper/base_wrapper.py`
的 `apply_freeze()`），在 DDP wrap **之前**的 `setup()` 里应用。三条语义：

1. **glob 匹配 + `!` 取反，后命中者胜**（`fnmatchcase(name, kw)`，不是前缀也不是正则）。
   裸子串要写成 `*patch_embed*`——**裸写 `patch_embed` 匹配不到任何东西**，会撞上第 3 条硬错。
   前缀 `!` 的关键词**取消选中**它匹配的名字：关键词按顺序读，最后一个命中的说了算，
   所以 `[*global_blocks*, !*.lora.*]` 读作"global blocks，但不含它们的 LoRA"。
   `*patch_embed*` 会命中 `part_head` 里同名子串这类事仍要靠 lock 的 diff 看，不靠想。
2. **只冻不解冻（增量式）**：选中置 `requires_grad=False`，未选中的**保持建模时的状态不动**。
   早期版本是无条件赋值 `requires_grad = not matched`，会把构造期冻结不变量（LoRA 基座）
   悄悄解冻，基座 + adapter 一起训、LoRA 完全失效。
   **`!` 只收窄"选中什么"，它不解冻**——同一条不变量因此原样成立。
3. **关键词零命中直接 `raise`**，防拼错单词导致"以为冻了其实没冻"。`!` 关键词一视同仁：
   拼错的 `!` 同样危险（它本该放行的东西会被冻掉）。

`param_groups` 只管分组学习率，`lr_multiplier` 必须 > 0；想冻结就写进 `freeze_keywords`。
**判断"某参数是否在训"看该配方的 lock，不看注释、不看这份文档。**

### 6.2 仓库里其余四处 `requires_grad=False`——都不是入口

| 位置 | 性质 |
|---|---|
| LoRA（`src/model/segvggt/layers/lora.py`）冻它包裹的 linear | **构造期冻结不变量**。注入在 `Aggregator._apply_lora`，只包 attention 的 qkv/proj（+可选 MLP）。只冻住被包裹的 attention 基座（约 202M），同 block 的 MLP/LayerNorm（约 403M）仍全量可训——vendored 官方行为。想连它一起冻死要另写 `frame_blocks`/`global_blocks` 关键词。 |
| `patch_embed.mask_token`（`segvggt/models/aggregator.py`、`vggt/models/aggregator.py`） | **构造期冻结不变量**。AnySplat 路线的构造期不变量**只有它一个参数**。 |
| 教师/参考网 + CPU 卸载（`arch/anysplat.py` 的 `distill_*`、`segvggt_wrapper` 的 `_geo_target_from_teacher`、`loss_depth.py` 的 DepthAnything） | **显存管理**，不是训练期冻结。`requires_grad=False` 且 `param.data` 常驻 CPU，前向在 `torch.no_grad()` 里。 |
| 推理期 `model.eval()` + 全参 `requires_grad_(False)`（`anysplat_wrapper._test_step_*`、`eval_pose.py`、`scripts/instseg_infer.py`、`scripts/trace_instance_to_gaussians.py`） | **推理**，不是训练期冻结。 |

历史上存在过第五处——`freeze_backbone` / `freeze_module`（AnySplat 官方上游代码）——它是
`freeze_keywords` 的严格功能子集，**已彻底删除**，用它的配方全部迁到 `freeze_keywords`。

### 6.3 lock：启动时的实测校验

- **一份配方一份 lock**，路径 `config/experiment/locks/<X>.lock`，`<X>` = hydra 的
  `+experiment=<X>`（即 yaml 文件名）。**不用 `wandb.name` 当身份键**：26 份里 7 份与文件名不等、
  且有三组重名（`instseg_inscene_infinigen` ×2、`instseg_inscene_scannetpp_v2` ×2、`segvggt_finetune_agnostic` ×3），重名意味着数份配方共用一份 lock。
- **全覆盖 + 缺 lock = 启动硬错**（26/26）。不是"有就校验"——否则"这份配方没写 lock"和
  "这份配方不需要冻结"长得一样，而"忘了写 `freeze_keywords`"正是三种故障形态之一。
- **校验时机**：`BaseWrapper.setup()` 末尾、strategy wrap 之前，唯一的门是 `stage == "fit"`
  （`test` 下没有优化器，护栏守的是空气）。`fast_dev_run` 与 sanity check **零豁免**。
- **硬错三条**：正文逐行不符 / lock 缺失 / 关键词零命中。**只记录不执法两项**：`base_lr`
  与 `dtype`（进正文与哈希，但无专门断言——改了哈希自然会撞，人被迫看一眼）。**没有 warn 档。**
- 校验比对的是 lock 的**正文**，不是 lock 头部自己声明的 `structure:` 哈希——否则手改正文
  不改哈希行就能全放行，"人签字的文本"与"被执法的文本"就不是同一份了。
- 失败时把实测指纹全文写进 run 目录并在报错里点名（experiment / lock 路径 / 重生成命令 /
  头几行 diff）。mismatch 常发生在远程节点上，若差异来自环境，本地重跑复现不出来。
- 每个 rank 各算各的：纯 CPU walk，而 rank 之间冻结不一致是最难从 loss 曲线上看出来的故障。

### 6.4 改配方时人要做的五步

1. 改 `config/experiment/<X>.yaml` 的 `freeze_keywords`（或改动了任何影响参数集的结构）。
2. `python scripts/freeze_lock.py +experiment=<X>` 重新生成 lock（纯 CPU，不加载权重，
   不需要 GPU 节点，也不需要集群上的 ckpt 路径）。
3. **读 `git diff`**：第一列是 `requires_grad`（`T`/`-`），第二列是 dtype。
   `5f1eff7` 那类漏冻（`camera_token`/`register_token`）在 diff 里就是多出来的两行。
4. **lock 与 yaml 一起提交**。git diff 就是签字现场——终端上没有第二道确认仪式。
5. 新增 arch / 新增 experiment 无需任何登记：缺 lock 是硬错，它跑不起来，直到有人生成并看过一份 lock。

批量重生成用 `python scripts/freeze_lock.py --all`：**一进程一份、串行子进程**。单份构建峰值
RSS 约 7.3G，在一个进程里连建 26 份会把机器抖死（swap 抖死，不是 OOM kill）。

### 6.5 不由这一层负责的事

- **`lr` / `param_groups` 的判等**：`lr` 不占冻结三判据的任何一条。`base_lr` 只记进 lock 头部
  （不参与哈希、不判等）保留可见性；"注释里的 base lr 过期了"属配方审查。
- **跨 stage 的冻结关系**：全仓只有一条真正的 stage 链边（`segvggt_physgm` ←
  `segvggt_finetune_agnostic` 的产物），实测自洽。对链断裂的防线是 lock 的 `git diff`。
- **bf16 死参数**：见 `CONTEXT.md` 同名词条——属训练精度策略，机制在另一层。
- **权重落位断言**（`_assert_query_physgm_coverage`）：查的是权重的**值**是否加载成功，
  与冻结三判据无关，留在原地不并进本层。


## 7. scripts 与 trace 管线（下游推理/导出）

- `scripts/instseg_infer.py`：统一推理脚本（两种输入模式 × 多种输出：seg2d/pca2d/seg3d_ply/embedding/video…），聚类算法 kmeans/hdbscan 在 `src/instseg/` 里。
- `scripts/trace_instance_to_gaussians.py`：把 2D 特征图 trace 到已训好的 2DGS/3DGS 高斯上（依赖带 `trace()` 的 diff_[surfel|gaussian]_rasterization CUDA 扩展，`TRACE_CHANNELS` 必须与 `src/trace_render/trace_rasterize.py` 一致）。特征来源可选 anysplat / iggt / precomputed / gt_idmap。
- `scripts/export_seg3d_vlm_views.py` + `src/trace_render/`：把 seg3d_split 聚类 PLY 逐实例渲染成 PNG 树给 VLM 用（configs/vlm_export/ 的 job JSON 驱动）。
- `src/trace_cameras/`：trace 脚本专用的相机加载（COLMAP / transforms.json，注册表 + `--trace_config`），与训练侧数据管线**独立**，别互相混用。
- 根目录 `iggt_idmap.py`：多视角 IGGT 特征 → HDBSCAN 跨视角聚类 ID map（可选 3D KNN 平滑，pyg/scipy 两种后端）。



## 8. 已知的坑 / AI 常犯错误清单

1. **训练数据管线统一在** `src/dataset/`。`src/instseg/` 只保留推理/trace 后处理工具（kmeans / hdbscan_assign / export 等），不要在这里新增 dataset/datamodule 副本。
2. **heads 按领域分组**：自研 head 统一在 `src/model/heads/{gaussian,instance,physics}/`；vendored VGGT 自带的 camera/dpt/track head 留在 `src/model/vggt/heads/` 盒子里，两边不要互相搬。新 head 按领域入组，或新建领域子目录。
3. `EncoderIGGT.forward` 里对 `instance_feat_map` 有**硬编码 L2 normalize**（`iggt.py` 有注释 "hard code normalize for iggt"）；而 disc loss 又"没按原文做 L2 归一化"——改归一化策略时两处要一起考虑，别重复归一化。
4. `EncoderAnySplat` 的 instance head 由 `instance_feat_dim` 控制（0 = 禁用，`config/model/encoder/anysplat.yaml` 默认 0）；IGGT 默认 8。同一个 PartHead 被两个 encoder 共享——这正是"同一 head 换 backbone"的实验入口，**改 PartHead 接口时两个 encoder 都要过一遍**。
5. 参数冻结唯一入口是 `optimizer.freeze_keywords`，语义是**只冻不解冻**（增量式）、glob 匹配 + `!` 取反（后命中者胜）、零命中 `raise`。**每份 experiment 配方必须带一份 lock**（`config/experiment/locks/<X>.lock`），启动时逐行校验实测的可训集合，不符或缺失即拒训——改了 `freeze_keywords` 就要重生成 lock 并一起提交。判断"某参数是否在训"**看它的 lock**，不看注释。机制全文见 §6，术语见根 `CONTEXT.md`。
6. 精度约定：camera/point/depth head 强制 fp32，loss 计算强制 fp32（`autocast enabled=False`）。别"顺手统一"精度。**注意 aggregator 不只是 autocast**：`arch/anysplat.py:113` 与 `arch/iggt.py:80` 对 aggregator 做了**无条件的永久 `.to(torch.bfloat16)`**（`arch/segvggt.py` 干净，只用 autocast）。后果是它的 AdamW 动量状态也是 bf16，量级 ~lr 的更新被舍成 0——参数 `requires_grad=True` 却不再更新。这**不是冻结**（见根 `CONTEXT.md` 的"bf16 死参数不是冻结"），是训练精度问题；22 份配方里有 5 份带着可训的 bf16 aggregator 参数（`instseg_small` 约 909M，占其可训参数的 72.8%）。
7. Hydra 配置是**类型化的**（`src/config.py` `load_typed_root_config` + dataclass + beartype/jaxtyping import hook）。加配置项必须同步改对应 cfg dataclass，否则启动即报错；jaxtyping 的 shape 标注是运行时校验，改张量布局时记得改标注。
8. 历史上已做过的清理，不要走回头路：post_opt 已全删；blender2opencv 手写矩阵已清理（统一走 src/coord）；trace 相机加载已抽到 src/trace_cameras 注册表。
9. AnySplat 原始遗留死代码已于 2026-07 按引用图证据清理（`encoder/backbone/` croco/dino/resnet 全树、`model/transformer/`、`model/encodings/`、epipolar visualizer、`decoder/cuda_splatting.py`、`vggt/utils/visual_track.py`、`utils/ba.py`、`loss_point.py`、`validation_in_3d.py`、`ptc_geometry.py`）。仍保留的"看似没用"代码只有一处是刻意的：`vggt/heads/track_head.py` + `track_modules/`（`VGGT.__init__` 无条件构造，删了会破坏 HF 权重加载）。今后删代码前先做引用图核查（含函数内惰性导入），有证据即可删，git 历史兜底。



## 9. 当前活跃工作区

- 主要改动集中在：`src/model/`、`src/trace_*`、`scripts/`。
- 近期方向（git log）：seg3d 分割结果按实例分别渲染给 VLM；trace 支持自研 3DGS；idmap 的 3D KNN 加速。
- 实验配置见 §2 的算法登记表：IGGT 路线 `instseg_iggt*.yaml`，AnySplat 路线 `instseg_anysplat.yaml` / `instseg_inscene_*.yaml`，物理属性 `phys_iggt.yaml`。


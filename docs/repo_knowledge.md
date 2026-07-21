# 仓库现状理解（供 AI 阅读，避免常见误判）

> 本文档描述仓库的**真实现状**（探索版，非定稿），供 AI 开发时对齐认知。
> 与 `memory.md` 配合阅读：memory.md 记录项目事实，本文档记录架构约定与"坑"。
> **新增能力的分层写法**以 `[layered_scheme.md](layered_scheme.md)` 为准（本文件偏现状与坑）。

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
| 6   | SegVGGT 端到端实例（iter 1 推理 + iter 2 训练） | `segvggt_finetune_agnostic.yaml`（stage-1 class-agnostic 微调）；`segvggt_scannet.yaml`（论文全量训练）；`scripts/segvggt_infer.py`（推理）                                             | segvggt（encoder-only，object queries）；vendored box `src/model/segvggt/` + SemanticHead                          | manifest（instance_mask+相机+深度）  | —（class-agnostic 默认，可选 semantic） | segvggt（cls+BCE+Dice+FADA）, segvggt_geo（camera+depth） |


新增第 5 套算法时：新建 experiment yaml 组合三元组，并在本表加一行。

### SegVGGT 路线（新增，iteration 1 = 仅推理）

与 IGGT 不同：SegVGGT 把实例推理做进 transformer 内部——object queries 在每层 global attention 后 cross-attend 图像 token，每个 query 直接出「per-view mask + 类别分布（末通道 = no-object，天然排除背景/空 query）」，端到端、无聚类后处理、无 GT-mask 池化。这正是把 per-object 物理属性挂在 query 上的天然载体（iter 3 目标）。

- **vendored box**：`src/model/segvggt/`（改版 aggregator + CrossBlock/CrossAttention + SemanticHead + LoRA，忠实照搬官方，import 重写为 `src.model.segvggt.*`）。与现有 `src/model/vggt/` box 隔离，互不牵动。不搬 `dependency/`（VGGSfM tracker）与 `utils/geometry`。
- **arch**：`src/model/arch/segvggt.py`（`EncoderSegVGGTCfg` / `EncoderSegVGGT` 持有 vendored `SegVGGT` 为 `self.model`，forward 重打包成 `EncoderOutput`；`SegVGGTModel.from_checkpoint` 给官方 `.pt` 键加 `encoder.model.` 前缀后 shape 对齐 strict=False）。已注册进 `MODELS` / `EncoderCfg` union / `get_model`。
- **契约槽**：`EncoderOutput.segvggt_prediction`（`SegVGGTPrediction`：query_masks / query_class_logits / query_embed / feature_map / attn_frame_mean）。`query_embed`（per-object 嵌入）iter 1 暂不暴露（vendored forward 用完即弃），iter 3 需要时再接。
- **config**：`config/model/encoder/segvggt.yaml`（`enable_semantic: 20|200` 对应官方两套 ckpt；LoRA rank 32 必须与训练一致）。
- **权重**：官方 HuggingFace `JinyuanQu/SegVGGT`（`checkpoint/segvggt_scannet{v2,200}.pt`，各约 6.6GB，含 DINO backbone，非 `hf:` 前缀走 from_checkpoint）。**官方无训练代码**，loss（Hungarian + BCE/Dice + FADA JS + teacher 蒸馏）需 iter 2 民间复现。
- **验证到位（iter 1）**：本仓库构建的 state_dict 键集与官方 `SegVGGT`（同 eval 配置）**逐键一致（2606=2606，零差异）**→ 官方 ckpt 加载零 bad-missing/unexpected；随机权重 CPU 端到端小前向 shape 全部打通。真权重前向/掩码质量待集群跑（本机无 GPU、缺 gsplat/hydra，仅开发机）。

#### iteration 2 = 训练复现（官方无训练代码，对照论文民间复现）

论文（`~/tmp/move_segvggt/segvggt/SegVGGT.md` + `mental-model.md`）给了完整训练配方：`L_total = L_geo + L_inst + λ_js·L_js`。官方只发权重不发训练码，故按 IGGT 的 disc/mvc 先例民间复现。**全部落在 loss 层（纯公式，无可学习参数）+ wrapper 层**，backbone box 与 arch 契约零改动。

- **matcher**：`src/loss/segvggt_matcher.py`（纯 torch+scipy，无仓库依赖，可单测）。DETR/Mask2Former 式二部图匹配，cost = `-λ_cls·c_{j,ck} + λ_mask·(BCE+Dice) + λ_js·C_js`（论文 Eq.5）。多视角 mask 摊平成 `N·H·W` 即等价点云 mask，直接继承点云分割的 BCE+Dice。`C_js`=帧级注意力 JS 散度（Eq.8）。
- **实例 loss**：`src/loss/loss_segvggt.py`（`LossSegVGGT`，注册名 `segvggt`）。跑 matcher → `λ_cls·CE(带 no-object 通道) + λ_mask·(BCE+Dice) + λ_js·L_js(FADA)`。GT 全部从多视角 `instance_mask` 现推：每实例二值 mask（area 降采到预测分辨率）、帧可见度分布（面积占比，用于 FADA）、类别（默认 class-agnostic；给 `instance_semantic` 逐像素图且 `class_agnostic=false` 则多数投票转类别监督）。
- **id 0 的语义对本 loss 是硬约束（和 IGGT 不同）**：IGGT 的 `loss_disc`/`loss_mvc` 是对比型，只拉近/推远同异 id 像素，对 id 0 **不产生任何梯度**，所以 id 0 是"背景"还是"未标注"无所谓。SegVGGT 是集合预测：匹配上的 query 其 mask BCE/Dice 覆盖整帧，id-0 像素成为**负样本**——若 id 0 里藏着没标注的物体，训练会主动把它们压掉，比预训练偏差更糟。因此 `instance_valid_mask` 的无效像素**不能折成 id 0**（IGGT 那样折对对比 loss 无害，对这里有害），而要走真正的 ignore：matcher 和 loss 的 BCE/Dice 都接受逐像素 `keep` 权重，被忽略的像素既不是正样本也不是负样本（`keep` 必须两侧一致，否则匹配和监督对"哪些像素算数"意见不合）。配置开关 `loss.segvggt.unlabeled_as_ignore`（默认 false）可把 id-0 也一并划入 ignore；**注意**开了之后负样本只剩其他实例的像素，若 id 0 占比很大，mask 会失去收紧压力而膨胀。判定用 `scripts/check_instance_id0.py`（按连通域形状分辨：真背景=一个贴边的大连通域；未标注物体=若干位于内部的紧致连通域，看 "INTERIOR id-0 fraction"）。InsScene 三个源（infinigen / scannetpp_v2 / re10k）预处理管线不同，**必须分别测**。
- **class-agnostic 的正确做法 = 边缘化，不是"打成 class 0"**：本仓库 manifest 数据集只有多视角一致 instance id，没有语义标签。若把 GT 全打成 class 0，等于逼预训练头把 channel 0（ScanNet 的 "cabinet"）改造成"任意物体"、同时压掉另外 17 个通道，预训练的 objectness 先验被丢光。正解是对类别分布做边缘化：`P(object) = Σ_c P(c) = 1 - P(no-object)`，实现为在 `[logsumexp(前景 logits), no-object logit]` 上做二分类 CE（matcher 的 `cost_cls` 同步变成 `-P(object)`）。好处：预训练分类头**一个权重都不用改**（无 head surgery、ckpt 键不变）、与推理侧 `1-P(no-object)` 判据（`scripts/segvggt_infer.py`、wrapper 的 `val/queries_fired`）自洽、18 个前景通道的语义结构保留下来供 iter 3 的 per-query 物理读出使用。**推论：预训练权重在 init 时就已经是 class-agnostic 正确的**，所以 stage-1 微调的起始 loss 就应该不高——这本身是训练实现是否接对的一个体检点。
- **FADA 关键接线**：vendored forward 已出 `attn_frame_mean`（`[L,B,Q,S]`，= 每帧注意力质量的 `1/P·Σ`，即论文 `p̂` 差一个逐帧常数 P）；loss 里**沿帧维重归一化**即恢复论文归一化分布。FADA 双角色：既是 matcher 的 cost 项（`cost_js>0`），又是 matched pair 的正则 loss（`λ_js`）——两者共享一次匹配，故合在同一个 Loss 内算。
- **几何 loss**：`src/loss/loss_segvggt_geo.py`（`LossSegVGGTGeo`，注册名 `segvggt_geo`）。`λ_camera·L_camera + λ_depth·L_depth`。camera=9 维 pose encoding 上的 Huber（沿 camera head 迭代 γ 衰减，复用 `loss_huber`）；depth=按序列单一 median scale 对齐后 L1+梯度（VGGT 上到尺度）。监督源解耦：wrapper 经 `depth_dict["segvggt_geo_target"]` 喂 target——默认 `gt`（manifest 干净相机/深度，恒可跑），可选 `teacher`（冻结 VGGT 蒸馏，论文原味，需 1B 权重+全环境，`train.segvggt_geo_supervision: teacher` 开）。
- **wrapper**：`src/model/wrapper/segvggt_wrapper.py`（`SegVGGTWrapper`，镜像 `IGGTWrapper`，encoder-only 无渲染）。context+target 视角拼一起送 encoder，把 `segvggt_prediction`/`instance_mask`/`pred_pose_enc_list`/`depth`/`geo_target` 塞进 `depth_dict` 交给 loss。`src/main.py` 分发新增 `isinstance(EncoderSegVGGTCfg)→SegVGGTWrapper`（在 IGGT 分支之前）。
- **超参对齐论文 A.4**：`λ_camera=5, λ_depth=1, λ_cls=0.5, λ_mask=1, λ_js=0.5`，matching cost 权重=loss 权重；新参数 lr `2e-4`、pre-existing `6e-5`（=6e-5×3.333 param_group）、DINO(`patch_embed`) 冻结、LoRA rank 32、梯度裁剪。experiment：`config/experiment/segvggt_scannet.yaml`。
- **验证到位（iter 2）**：CPU 合成张量端到端跑通 matcher（二部图分配合法、cost 有限、JS(p,p)=0）+ `LossSegVGGT`（cls/BCE/Dice/FADA 四项有限、matched 计数正确、梯度回传到 query mask/类别 logits、空场景→仅 no-object CE、`cost_js=0` 退化为 loss-only FADA、class-agnostic 边缘化后梯度覆盖全部 18 个前景通道且 `P(obj)=1-P(no-obj)` 恒等式精确成立、class-aware 回退路径可跑）+ `LossSegVGGTGeo`（camera/depth 有限、梯度回传、缺 target→0）。真数据训练/收敛待集群（本机无 GPU）。

**stage-1 class-agnostic 微调**（`config/experiment/segvggt_finetune_agnostic.yaml`）——拿官方 ckpt 直接适配成 class-agnostic，用来体检训练实现：

- 冻结集 `[patch_embed, frame_blocks, global_blocks, camera_head, depth_head]`，即整个 aggregator（含 ckpt 里已训好的 LoRA）+ 几何头全冻；只训 instance 分支 + 实例分类头 = **487M 可训**（AdamW 动量约 3.9GB）。几何不可能漂移，故 `segvggt_geo.weight: 0`。
- 因为没有任何随机初始化的新参数（是适配不是从头训），论文的 `2e-4` 太烫，改单一 `lr: 2e-5`。
- 对照：论文全量配方 `segvggt_scannet.yaml` 冻结集只有 `[patch_embed]`，可训 **1148M**（AdamW ~9.2GB）。注意 LoRA 只冻住被包裹的 attention 基座（202M），frame/global block 的 MLP/norm（403M）仍全量可训——这是 vendored 官方代码的行为，不是我们加的。
- 实测参数量核对（`paper_repo` env CPU 建模）：`patch_embed`→344 个/304M（DINO）、`instance_`→1011 个/454M、`semantic_head`→62 个/32.6M、`lora_`→192 个/9.44M（96 层 × qkv+proj）。yaml 里的 keyword 全部命中，无空组。

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
- 图像张量约定：dataset 输出 `[-1, 1]`（normalize shim），但 encoder 吃 `[0, 1]`——wrapper 里有 `(image + 1) / 2`。改 wrapper/推理脚本时别弄丢这一步。
- IGGT 训练时 context + target 视角**拼在一起全部送入 encoder**（`torch.cat([context, target], dim=1)`），instance_mask 也对应拼接。



## 6. scripts 与 trace 管线（下游推理/导出）

- `scripts/instseg_infer.py`：统一推理脚本（两种输入模式 × 多种输出：seg2d/pca2d/seg3d_ply/embedding/video…），聚类算法 kmeans/hdbscan 在 `src/instseg/` 里。
- `scripts/trace_instance_to_gaussians.py`：把 2D 特征图 trace 到已训好的 2DGS/3DGS 高斯上（依赖带 `trace()` 的 diff_[surfel|gaussian]_rasterization CUDA 扩展，`TRACE_CHANNELS` 必须与 `src/trace_render/trace_rasterize.py` 一致）。特征来源可选 anysplat / iggt / precomputed / gt_idmap。
- `scripts/export_seg3d_vlm_views.py` + `src/trace_render/`：把 seg3d_split 聚类 PLY 逐实例渲染成 PNG 树给 VLM 用（configs/vlm_export/ 的 job JSON 驱动）。
- `src/trace_cameras/`：trace 脚本专用的相机加载（COLMAP / transforms.json，注册表 + `--trace_config`），与训练侧数据管线**独立**，别互相混用。
- 根目录 `iggt_idmap.py`：多视角 IGGT 特征 → HDBSCAN 跨视角聚类 ID map（可选 3D KNN 平滑，pyg/scipy 两种后端）。



## 7. 已知的坑 / AI 常犯错误清单

1. **训练数据管线统一在** `src/dataset/`。`src/instseg/` 只保留推理/trace 后处理工具（kmeans / hdbscan_assign / export 等），不要在这里新增 dataset/datamodule 副本。
2. **heads 按领域分组**：自研 head 统一在 `src/model/heads/{gaussian,instance,physics}/`；vendored VGGT 自带的 camera/dpt/track head 留在 `src/model/vggt/heads/` 盒子里，两边不要互相搬。新 head 按领域入组，或新建领域子目录。
3. `EncoderIGGT.forward` 里对 `instance_feat_map` 有**硬编码 L2 normalize**（`iggt.py` 有注释 "hard code normalize for iggt"）；而 disc loss 又"没按原文做 L2 归一化"——改归一化策略时两处要一起考虑，别重复归一化。
4. `EncoderAnySplat` 的 instance head 由 `instance_feat_dim` 控制（0 = 禁用，`config/model/encoder/anysplat.yaml` 默认 0）；IGGT 默认 8。同一个 PartHead 被两个 encoder 共享——这正是"同一 head 换 backbone"的实验入口，**改 PartHead 接口时两个 encoder 都要过一遍**。
5. 参数冻结唯一入口是 `optimizer.freeze_keywords`（`BaseWrapper.setup`，在 DDP wrap 前应用；关键词零匹配直接报错）。语义是**只冻不解冻**（增量式）：命中关键词的置 `requires_grad=False`，未命中的**保持建模时的状态不动**。这点很关键——有些模块在构造时就已冻结部分参数（如 LoRA 会冻掉它包裹的 attention 基座权重），早期版本这里是无条件赋值 `requires_grad = not matched`，会把这些参数**悄悄解冻**，导致基座+adapter 一起训练、LoRA 完全失效。`param_groups` 只管分组学习率，`lr_multiplier` 必须 > 0，想冻结就写进 `freeze_keywords`。判断"某参数是否在训练"看 freeze_keywords + arch 加载日志。例外：`EncoderAnySplat` 自带 `freeze_backbone`/`freeze_module`，仅在 freeze_keywords 为空时生效。
6. 精度约定：VGGT aggregator 跑 bf16 autocast，camera/point/depth head 强制 fp32，loss 计算强制 fp32（`autocast enabled=False`）。别"顺手统一"精度。
7. Hydra 配置是**类型化的**（`src/config.py` `load_typed_root_config` + dataclass + beartype/jaxtyping import hook）。加配置项必须同步改对应 cfg dataclass，否则启动即报错；jaxtyping 的 shape 标注是运行时校验，改张量布局时记得改标注。
8. 历史上已做过的清理，不要走回头路：post_opt 已全删；blender2opencv 手写矩阵已清理（统一走 src/coord）；trace 相机加载已抽到 src/trace_cameras 注册表。
9. AnySplat 原始遗留死代码已于 2026-07 按引用图证据清理（`encoder/backbone/` croco/dino/resnet 全树、`model/transformer/`、`model/encodings/`、epipolar visualizer、`decoder/cuda_splatting.py`、`vggt/utils/visual_track.py`、`utils/ba.py`、`loss_point.py`、`validation_in_3d.py`、`ptc_geometry.py`）。仍保留的"看似没用"代码只有一处是刻意的：`vggt/heads/track_head.py` + `track_modules/`（`VGGT.__init__` 无条件构造，删了会破坏 HF 权重加载）。今后删代码前先做引用图核查（含函数内惰性导入），有证据即可删，git 历史兜底。



## 8. 当前活跃工作区

- 主要改动集中在：`src/model/`、`src/trace_*`、`scripts/`。
- 近期方向（git log）：seg3d 分割结果按实例分别渲染给 VLM；trace 支持自研 3DGS；idmap 的 3D KNN 加速。
- 实验配置见 §2 的算法登记表：IGGT 路线 `instseg_iggt*.yaml`，AnySplat 路线 `instseg_anysplat.yaml` / `instseg_inscene_*.yaml`，物理属性 `phys_iggt.yaml`。


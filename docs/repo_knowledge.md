# claude_repo_knowledge.md — 仓库现状理解（供 AI 阅读，避免常见误判）

> 本文档描述仓库的**真实现状**（探索版，非定稿），供 AI 开发时对齐认知。
> 与 `memory.md` 配合阅读：memory.md 记录项目事实，本文档记录架构约定与"坑"。
> **新增能力的分层写法**以 [`layered_scheme.md`](layered_scheme.md) 为准（本文件偏现状与坑）。

## 1. 这个仓库是什么

- 起点是 fork 的 **AnySplat**（3DGS 前馈重建），目标是在其上做**语义/实例分割推理**等新东西。
- 后来发现 AnySplat backbone 效果不佳，而 **IGGT**（输入输出相似的工作）更合适，于是把 IGGT 的模型代码**抠进本仓库复用**，共享已写好的数据集载入、训练框架、推理脚本。
- 所以本仓库是**多个仓库的缝合体**：AnySplat 框架（Hydra + Lightning + 数据管线）+ VGGT backbone + IGGT 的 instance head（SamProjector + PartHead）+ 自研的 PhysicsHead、trace 管线、坐标工具。
- **这是探索版本，不是定稿**。owner 要频繁尝试：不同 backbone × 不同 head × 不同 loss 的组合。**层与层的隔离是第一设计原则**——改一个小模块不应牵动其他层。
- **分层方案契约（新增能力默认遵守）**：见 [`layered_scheme.md`](layered_scheme.md)。Physics 已按该规范落地；instance 等旧路径尚未完全迁移。

关键事实（来自 memory.md）：
- 训练基于 AnySplat 预训练权重（重建部分已优化好），只有 instance head 等新增结构随机初始化。
- IGGT 官方**没有开源训练代码**，所以 loss（mvc / disc）是民间复现，可能有误。mvc 是像素级对比 loss，很难优化；disc 是实例级判别 loss，分单视角/多视角，加了 soft hinge，没按原文做 L2 归一化和 L2 正则。

## 2. 分层架构（最重要的一节）

从底到顶四层，**只允许上层依赖下层**：

```
底层部件   src/model/encoder/  src/model/decoder/
           ├─ encoder/vggt/          VGGT backbone（aggregator + camera/point/depth head）
           ├─ encoder/backbone/      AnySplat 原有 backbone（croco/dino/resnet）
           ├─ encoder/heads/         AnySplat 的 GS head（DPT 等）
           ├─ encoder/iggt_heads/    SamProjector、PartHead、PhysicsHead、PhysicsClassifier
           ├─ encoder/anysplat.py    EncoderAnySplat：backbone + GS head（可选挂 instance head）
           ├─ encoder/iggt.py        EncoderIGGT：VGGT + PartHead（可选 PhysicsHead+Classifier），无 GS head
           └─ decoder/               splatting CUDA 渲染 decoder（目前只有 AnySplat 用）

arch 层    src/model/arch/     负责"最终组装模型"
           ├─ anysplat.py           AnySplat = encoder + decoder
           ├─ iggt.py               IGGTModel = 纯 encoder（无 decoder）+ 官方 ckpt 键名重映射加载
           └─ __init__.py           get_model(encoder_cfg, decoder_cfg)：按 cfg 类型分发 + 预训练权重加载逻辑

wrapper 层 src/model/           负责"再包一层训练"，对接 Lightning 训练/推理
           ├─ base_wrapper.py       BaseModelWrapper：optimizer 分组、loss 汇总、日志、可视化等公共逻辑
           ├─ anysplat_wrapper.py   渲染重建训练（decoder 渲染 → mse/lpips/depth 等）
           └─ iggt_wrapper.py       encoder-only 训练（instance/physics 特征 → disc/mvc/phys loss），无渲染

入口       src/main.py          Hydra 入口；scripts/ 下各推理/导出脚本
```

**统一的层间契约**：所有 encoder 的 forward 返回 `EncoderOutput`（`src/model/encoder/encoder.py`），字段包括 `gaussians`（IGGT 为 None）、`pred_context_pose`、`depth_dict`、`instance_feat_map [B,V,N,H,W]`、`gaussian_instance_feat`、`physics_prediction`（`PhysicsPrediction | None`）。**新增 head 输出时，往 EncoderOutput 加可选字段（默认 None），不要改已有字段语义**——这是 wrapper 与 loss 之间解耦的接口。

**分发点（改组合时要看的三个注册表）**：
1. `src/model/arch/__init__.py` 的 `MODELS` / `get_model`：按 `encoder_cfg` 的 dataclass 类型分发到 arch。
2. `src/main.py` L128：按 `isinstance(cfg.model.encoder, EncoderIGGTCfg)` 选 `IGGTWrapper` 还是 `AnySplatWrapper`。
3. `src/loss/__init__.py` 的 `LOSSES` 字典 + `src/dataset/__init__.py` 的 `DATASETS` 字典。

新增 backbone/arch/head/loss 时，照抄现有 iggt 的接线方式：新 cfg dataclass → 注册到对应字典/Union 类型 → 加 `config/model/encoder/` 或 `config/loss/` 下的 yaml → 用 `config/experiment/` 组合。

## 3. 权重加载（按来源分流，权威在 arch）

实现集中在 [`src/model/arch/weight_loading.py`](../src/model/arch/weight_loading.py) + [`get_model`](../src/model/arch/__init__.py) + [`IGGTModel.from_checkpoint`](../src/model/arch/iggt.py)。**scripts 禁止再复制** `_load_lightning_ckpt` / HF `strict=False` 白名单；新脚本只调这些入口。

四类旋钮（不要混成一个假统一 API）：

| 阶段 | 旋钮 | 权威实现 |
|------|------|----------|
| 建模时灌预训练 | `encoder.pretrained_weights` | `get_model` → HF 走 `init_anysplat_from_hf`；IGGT 本地路径走 `IGGTModel.from_checkpoint` |
| 训练 resume（含 optimizer/step） | `checkpointing.load` | Lightning `Trainer.fit(ckpt_path=...)`（`src/main.py`） |
| 推理加载训后权重 | `--ckpt` + `run_dir` | `load_model_from_run` |
| 快速 demo（无 run_dir） | 脚本 `--hf_model` | `init_anysplat_from_hf`（与 `get_model` 共用白名单） |

「手里有什么 → 用哪个」：

| 手里有什么 | 用哪个旋钮 |
|------------|------------|
| HF 发布的 AnySplat / 配置里 `hf:...` | `encoder.pretrained_weights`（训练）或 `--hf_model`（无 run 的脚本） |
| IGGT 官方/本地 `.pth`（需键名 remap） | `encoder.pretrained_weights`（非 `hf:` 前缀） |
| 自己训出的 Lightning `.ckpt` | 训练 resume → `checkpointing.load`；推理 → `--ckpt` + `run_dir` |

细节：

- **AnySplat HF**：`init_anysplat_from_hf` 加载后 `strict=False` 灌入；新增 head 的 missing keys 靠 `ALLOWED_ANYSPLAT_MISSING_PREFIXES`（`encoder.instance_head.` / `encoder.part_adaptor.` / `encoder.part_head.` 等）放行。**新增 head 后必须把前缀加进该常量**，不要在脚本里另写一份。计数与 bad_missing 前缀同样会 print（见上）。
- **IGGT 官方 ckpt**：`IGGTModel.from_checkpoint` 做两步键名重映射（`part_head.scratch.X → part_head.X`；补 `encoder.` 前缀），再按 shape 对齐后 `strict=False`。
- **VGGT backbone**：encoder 构造时 `VGGT.from_pretrained("facebook/VGGT-1B")`，是构造副作用，不是用户旋钮。
- **Lightning `.ckpt`**：`load_lightning_state_dict` 剥 `state_dict`；`load_model_from_run` 读 `run_dir/.hydra/config.yaml` → `get_model` → Wrapper → `load_state_dict`，返回 `wrapper.model`（只要 encoder 则取 `.encoder`）。`strict=False` 的 missing/unexpected 计数会 **print 到 stdout**（不只靠 logger），因为推理脚本通常未配 logging。


## 4. loss 体系

- 基类 `src/loss/loss.py`：`forward(prediction: DecoderOutput, batch, gaussians, depth_dict, global_step)`。
- **约定（有点脏但是现状）**：IGGT 路线没有 decoder 输出，`IGGTWrapper.training_step` 把监督信号塞进 `depth_dict` 传给 loss（`prediction`/`gaussians` 传 None）。instance 路线仍用裸 key（`instance_feat_map` / `instance_mask`）；**physics 路线用结构化对象**：`physics_prediction`（`PhysicsPrediction`）+ `physics_target`（`list[PhysicsTarget]`）。
- 分割相关 loss：`loss_disc.py`（实例判别，主力）、`loss_mvc.py`(像素对比，难优化)、`loss_phys.py`（物理属性分类——**纯公式，无可学习参数**；classifier 在 encoder 的 `PhysicsClassifier`）。
- loss 的开关和权重完全由 Hydra 的 `loss: [disc]` 列表 + `config/loss/*.yaml` 控制，代码里没有 if 开关。

Physics / 通用分层契约、改需求指哪里、反模式：见 [`layered_scheme.md`](layered_scheme.md)（权威）；Cursor rule：`.cursor/rules/layered-scheme.mdc`。

## 5. 数据管线

- 主力数据集是 `src/dataset/dataset_custom.py`（`name: custom`）：**manifest.jsonl 驱动**的通用多视角数据集（rgb / depth / instance_mask / K_px / c2w），InsScene-15K 各子集（infinigen / scannetpp / re10k）都走它。manifest 由 `scripts/make_manifest_*.py`、`scripts/extract_and_make_manifest_insscene.py` 生成。
- Physics 监督：`dataset.custom.physics_parser` 指向 `src/dataset/physics/parsers.py` 注册表；产出顶层 `batch["physics_target"]`（`list[PhysicsTarget]`，collate 在 `src/dataset/collate.py`）。
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

1. **训练数据管线统一在 `src/dataset/`**。`src/instseg/` 只保留推理/trace 后处理工具（kmeans / hdbscan_assign / export 等），不要在这里新增 dataset/datamodule 副本。
2. **两套 heads 目录**：`encoder/heads/` 是 AnySplat 原有 GS head，`encoder/iggt_heads/` 是 IGGT 抠来的 instance head。新分割/属性 head 放 `iggt_heads/` 或新建目录，别混进 `heads/`。
3. `EncoderIGGT.forward` 里对 `instance_feat_map` 有**硬编码 L2 normalize**（`iggt.py` 有注释 "hard code normalize for iggt"）；而 disc loss 又"没按原文做 L2 归一化"——改归一化策略时两处要一起考虑，别重复归一化。
4. `EncoderAnySplat` 的 instance head 由 `instance_feat_dim` 控制（0 = 禁用，`config/model/encoder/anysplat.yaml` 默认 0）；IGGT 默认 8。同一个 PartHead 被两个 encoder 共享——这正是"同一 head 换 backbone"的实验入口，**改 PartHead 接口时两个 encoder 都要过一遍**。
5. IGGT encoder 里 backbone 冻结（`freeze_backbone: true`），且 optimizer 还有 `freeze_keywords` / `param_groups`（`base_wrapper.configure_optimizers`）双重控制。判断"某参数是否在训练"要同时看这两处 + arch 加载日志。
6. 精度约定：VGGT aggregator 跑 bf16 autocast，camera/point/depth head 强制 fp32，loss 计算强制 fp32（`autocast enabled=False`）。别"顺手统一"精度。
7. Hydra 配置是**类型化的**（`src/config.py` `load_typed_root_config` + dataclass + beartype/jaxtyping import hook）。加配置项必须同步改对应 cfg dataclass，否则启动即报错；jaxtyping 的 shape 标注是运行时校验，改张量布局时记得改标注。
8. 历史上已做过的清理，不要走回头路：post_opt 已全删；blender2opencv 手写矩阵已清理（统一走 src/coord）；trace 相机加载已抽到 src/trace_cameras 注册表。
9. 本仓库有很多 AnySplat 原始遗留（`src/model/encoder/backbone/`、`heads/`、evaluation、visualization 的部分文件），当前 IGGT 分割路线**不经过它们**。不要因为"看起来没用"就删除，也不要误以为它们在当前训练路径上。

## 8. 当前活跃工作区

- 主要改动集中在：`src/model/`、`src/trace_*`、`scripts/`。
- 近期方向（git log）：seg3d 分割结果按实例分别渲染给 VLM；trace 支持自研 3DGS；idmap 的 3D KNN 加速。
- 实验配置看 `config/experiment/instseg_iggt_*.yaml`（IGGT 路线）和 `instseg_inscene_*.yaml` / `instseg_custom.yaml`（AnySplat 路线）；物理属性实验 `phys_iggt_custom.yaml`。

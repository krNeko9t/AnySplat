## 仓库理解与改造落地指南（Repo Understanding & Extension Brief）— AnySplat

> 目标导向：让你**快速知道“它在做什么、怎么跑起来、核心实现落在哪、复现路径是什么、扩展点在哪”**，并能把“论文改进想法”直接拆给 Cursor 做落地实现。  
> 仓库：AnySplat（Feed-forward 3D Gaussian Splatting from Unconstrained Views）

---

## 1) 代码结构地图（Architecture Map）

### 1.1 顶层入口（你真正会跑的脚本）
- **训练/测试主入口（Hydra + PyTorch Lightning）**：`src/main.py`
  - 负责：加载 Hydra 配置 → 构建 `AnySplat` 模型 → 包装成 `ModelWrapper` → 构建 `DataModule` → `trainer.fit/test`
- **推理（从 HuggingFace 拉模型，输出视频/PLY）**：`inference.py`（更干净）、`run.py`（偏临时脚本）
- **评估**
  - **NVS 指标**（PSNR/SSIM/LPIPS + 输出可视化）：`src/eval_nvs.py`
  - **CO3D Pose 指标**（AUC@{5,10,20,30}，可选 BA）：`src/eval_pose.py`
- **Demo（Gradio）**：`demo_gradio.py`
- **后优化（post-opt, 走 gsplat 的优化/增密策略）**：`src/post_opt/simple_trainer.py`

### 1.2 配置系统（Experiment System / 配置如何生效）
- **Hydra 根配置**：`config/main.yaml`
  - 定义：默认 encoder/decoder/loss、wandb、trainer、checkpointing、train/test 行为、`hydra.run.dir`
- **实验配置（你新增实验主要改这里）**：`config/experiment/*.yaml`
  - 示例：`co3d.yaml` / `dl3dv.yaml` / `scannetpp.yaml` / `multi-dataset.yaml`
  - 典型内容：`defaults` 组合、`wandb.name/tags`、`model.encoder.*`、`dataset.*`、`loss.*`、`trainer.*`、`hydra.run.dir`
- **数据集与 view sampler 配置组**：`config/dataset/*.yaml`、`config/dataset/view_sampler/*.yaml`
- **损失配置组**：`config/loss/*.yaml`
- **模型配置组**：`config/model/encoder/*.yaml`、`config/model/decoder/*.yaml`、`config/model/encoder/backbone/*.yaml`
- **类型化配置加载（DictConfig → dataclass）**：`src/config.py`
  - 关键点：`RootCfg` 里 `dataset`/`loss` 是 **list[Union wrapper]**，靠 `separate_*_cfg_wrappers()` 做 union 解包。

### 1.3 数据（Data）
- **Lightning DataModule**：`src/dataset/data_module.py`
  - 负责：根据 `cfg.dataset` 构建 dataset(s)；训练/验证用 `MixedBatchSampler` 做混合采样；返回 dataloaders
- **Dataset registry + 拼接逻辑**：`src/dataset/__init__.py`
  - `get_dataset(cfgs, stage, ...)`：train/val 返回 `CustomConcatDataset + datasets_ls`；test 返回 `TestDatasetWarpper`
- **Batch 类型定义（理解张量长相的最短路径）**：`src/dataset/types.py`
  - `BatchedExample = {context, target, scene}`；`BatchedViews` 包含 `image/extrinsics/intrinsics/near/far/index/overlap`
- **采样器（动态 batch/多数据集混合）**：`src/dataset/data_sampler.py`
  - `MixedBatchSampler` 内部用 `DynamicDistributedSampler` + `DynamicBatchSampler` 动态控制 **num_context_views / patch size(ps_h)** 等
- **View sampler（决定 context/target 选哪些帧）**：`src/dataset/view_sampler/*`
  - 注册：`src/dataset/view_sampler/__init__.py`
  - 常用：`view_sampler_bounded.py`（bounded gap）、`view_sampler_rank.py`（基于位姿距离排序挑帧）

### 1.4 模型（Model）
- **模型装配（硬编码 AnySplat）**：`src/model/model/__init__.py`
- **主模型类（HubMixin, inference/forward）**：`src/model/model/anysplat.py`
  - `inference(context_image)`：只跑 encoder，返回 `gaussians + pred_context_pose`
  - `forward(context_image)`：encoder → 用 predicted pose 直接 render（decoder）
- **Encoder（VGGT backbone + heads + voxelize + gaussian adapter）**：`src/model/encoder/anysplat.py`
  - 核心组件：
    - VGGT：`self.aggregator / self.camera_head / self.depth_head or point_head`
    - GS 参数头：`self.gaussian_param_head = VGGT_DPT_GS_Head(...)`
    - 可选 voxelize：`voxelizaton_with_fusion()`
    - 高斯参数映射：`GaussianAdapter` / `UnifiedGaussianAdapter`
- **Decoder（gsplat rasterization）**：`src/model/decoder/decoder_splatting_cuda.py`
  - `gsplat.rasterization(..., render_mode="RGB+D")` 输出 RGB+Depth+alpha

### 1.5 损失（Loss）
- **loss registry**：`src/loss/__init__.py`（`LOSSES` 映射 wrapper→实现类）
- **典型 loss 实现**
  - `src/loss/loss_mse.py`：支持 `mask/alpha/conf` 作为像素 mask
  - `src/loss/loss_lpips.py`：支持 `apply_after_step` + mask 逻辑
  - `src/loss/loss_depth_consis.py`：rendered depth vs VGGT depth 的一致性（可 EdgeAware / detach / mask）

### 1.6 训练循环/日志/可视化（Training + Logging + Viz）
- **LightningModule 包装层（训练/测试/验证都在这）**：`src/model/model_wrapper.py`
  - `training_step()`：组 batch → data_shim → model → metrics → sum(losses) → log
  - `configure_optimizers()`：AdamW + warmup + cosine；并做 **pretrained/new params 分组**（影响你改网络后的 LR 策略）
- **WandB / LocalLogger**
  - WandB：`src/main.py` 使用 `WandbLogger`
  - fallback：`src/misc/LocalLogger.py`（会 `rm -r outputs/local`）
- **评估指标实现**：`src/evaluation/metrics.py`

---

## 2) 核心数据流（Critical Data Flow）

### 2.1 Batch 从 DataLoader 出来长什么样（结构/shape）
以 `src/dataset/types.py` 为准（训练里实际使用的 key 在 `ModelWrapper` 可见）：

- **`batch: BatchedExample`**
  - **`batch["scene"]`**：`list[str]`
  - **`batch["context"]`**：多视角输入（训练里经常用它做重建/一致性监督）
    - `image`: `FloatTensor [B, Vc, 3, H, W]`
    - `extrinsics`: `[B, Vc, 4, 4]`
    - `intrinsics`: `[B, Vc, 3, 3]`
    - `near/far`: `[B, Vc]`
    - `index`: `[B, Vc]`
    - `overlap`: `[B, Vc]`（训练/测试里用于打 tag）
    - （实现里还出现了）`valid_mask`: `[B, Vc, H, W]`（用于 MSE/LPIPS mask）
  - **`batch["target"]`**：目标视角（test/eval 时用于 novel-view）
    - 同样字段，维度 `[B, Vt, ...]`

### 2.2 进入 Model 前经过了什么变换（关键的是“值域”）
- **Data shim（在 `ModelWrapper.__init__` 绑定）**：`self.data_shim = get_data_shim(self.model.encoder)`  
  - encoder 提供的 shim：`EncoderAnySplat.get_data_shim()` → `apply_normalize_shim()`  
  - `apply_normalize_shim`：把 `context.image` 做 `(x-mean)/std`（默认 mean/std=0.5）→ **把 [0,1] 变成 [-1,1]**
- **训练时显式再变回去**：`ModelWrapper.training_step()`  
  - `context_image = (batch["context"]["image"] + 1)/2` → **encoder 实际吃 [0,1]**

> 需要你跑一次确认的点：`validation_step()` 里调用 `self.model(batch["context"]["image"], ...)` **没有做 `(x+1)/2`**，和训练/推理脚本的约定不一致；若你依赖 validation 指标，建议优先 sanity-check 这段的数据值域是否符合 VGGT/encoder 预期。

### 2.3 forward 内关键子模块与特征维度（只抓主干）
以 `EncoderAnySplat.forward()` 为主：

1) **输入**
- `image: [B, V, 3, H, W]`（训练里是 `[0,1]`）

2) **VGGT aggregator**
- `aggregated_tokens_list, patch_start_idx = self.aggregator(image_bf16, intermediate_layer_idx=...)`
- `aggregated_tokens_list`：多层 token 列表（典型形态：`[(B*V), N_tokens, D]`）

3) **相机 head（pose）**
- `pred_pose_enc_list = self.camera_head(aggregated_tokens_list)`
- `pose_encoding_to_extri_intri(last_pose_enc, image.shape[-2:])` → `extrinsic, intrinsic`
- encoder 最后输出的 `pred_context_pose` 里：
  - `extrinsic`: 通过 padding + inverse 转成 `[B, V, 4, 4]`（c2w-like）
  - `intrinsic`: 做了按 `w/h` 的归一化（使 decoder 里可再“反归一化”）

4) **深度/点 head（几何先验）**
- `pred_head_type == "depth"`：`depth_map, depth_conf = self.depth_head(...)`
  - `pts_all = unproject(depth_map, extrinsic, intrinsic)` → `pts_all: [B, V, 3, H, W]`
- （另一分支）`pred_head_type == "point"`：直接预测点图 `pts_all`

5) **Gaussian 参数 head（每像素高斯的“原始参数/特征”）**
- `out = self.gaussian_param_head(tokens, pts_all(B*V), image, ...)`
- `anchor_feats = out[..., :raw_gs_dim]`，`conf = out[..., raw_gs_dim]`
- 随后：
  - **可选 voxelize**：`voxelizaton_with_fusion(anchor_feats, pts_all, voxel_size, conf)` → 每体素融合特征/点
  - 否则按 `conf_valid_mask` 选点/特征

6) **GaussianAdapter（最终 Gaussians）**
- `gaussians = self.gaussian_adapter.forward(neural_pts, depths, opacity, neural_feats)`
- 输出 `Gaussians`：`means/covariances/harmonics/opacities/scales/rotations`

7) **Decoder（渲染）**
- `DecoderSplattingCUDA.forward(gaussians, extrinsics, intrinsics, near, far, (H,W))`
- 内部对 intrinsics 做反归一化：`K[:,0]*=W`, `K[:,1]*=H`
- `gsplat.rasterization(..., render_mode="RGB+D")`
- 输出 `DecoderOutput`：
  - `color: [B, V, 3, H, W]`
  - `depth: [B, V, H, W]`
  - `alpha: [B, V, H, W]`

---

## 3) 实验与配置体系（Experiment System）

### 3.1 配置如何生效（Hydra 实际链路）
- 入口：`src/main.py` 的 `@hydra.main(config_path="../config", config_name="main")`
- 运行时配置 → 类型化：`cfg = load_typed_root_config(cfg_dict)`（`src/config.py`）
- 全局可读 config（很多地方直接读原始 DictConfig）：`src/global_cfg.py` 的 `set_cfg(cfg_dict)` + `get_cfg()`

### 3.2 如何新增一个实验（最短可复用套路）
你通常只需要：
- **新增** `config/experiment/<your_exp>.yaml`
- 然后运行：`python src/main.py +experiment=<your_exp> ...overrides...`

建议 `config/experiment/<your_exp>.yaml` 里包含：
- **`defaults`**：选择 dataset、encoder/decoder/backbone、loss 列表
- **`wandb.name/tags`**：决定实验名与归类
- **`hydra.run.dir`**：决定输出目录结构
- **`trainer.* / optimizer.* / checkpointing.* / loss.* / dataset.*`**：你要改的超参与结构开关

### 3.3 输出、日志、checkpoint 的位置（复现/对比实验必看）
- **Hydra 输出目录**：`config/main.yaml` 的 `hydra.run.dir`（默认 `output-debug/...`，实验里常 override 到 `output/exp_${wandb.name}/...`）
- `src/main.py` 会把 `cfg.train.output_path` 指向 `hydra.runtime.output_dir`
- **Lightning checkpoints**：`{output_dir}/checkpoints/`（`src/main.py` 的 `ModelCheckpoint`）
- **加载 checkpoint**
  - `checkpointing.load = <path>` 或 `wandb://<run_id>[:vX]`（`src/misc/wandb_tools.py`）
- **Test 输出**：`cfg.test.output_path`（默认 `outputs/test-nopo/<wandb.name>/...`，见 `ModelWrapper.test_step()`）

### 3.4 指标与评估脚本入口/输出格式
- **NVS**：`python src/eval_nvs.py --data_dir ... --output_path outputs/nvs`
  - 输出：`outputs/nvs/gt/*.jpg`, `outputs/nvs/pred/*.jpg` + 打印 mean PSNR/SSIM/LPIPS
- **CO3D Pose**：`python src/eval_pose.py --co3d_dir ... --co3d_anno_dir ... [--use_ba]`
  - 输出：`co3d_results_<timestamp>.txt`（AUC 汇总）
- **训练/测试中在线指标**
  - `ModelWrapper` 里 log：`train/psnr_probabilistic`, `val/psnr/lpips/ssim`, 以及各 `loss/*`

---

## 4) 扩展与改造的落点建议（Extension Points + 连锁影响）

下面按你常见改法给“落点清单 + 会影响的链路”。

### 4.1 改网络结构 / 新增模块（Encoder/Decoder/Head）
- **主要落点**
  - **Encoder 主干与 heads**：`src/model/encoder/anysplat.py`
    - 你可改：token 聚合策略、pose/depth 分支、`gaussian_param_head`、voxelize 融合、opacity mapping
  - **高斯参数映射（最常见扩展点）**：`src/model/encoder/common/gaussian_adapter.py`（你做“参数化方式/SH/scale/rot”等改动基本在这）
  - **Decoder 渲染/后处理**：`src/model/decoder/decoder_splatting_cuda.py`
- **连锁影响**
  - **optimizer 分组策略**：`ModelWrapper.configure_optimizers()` 只把名字包含 `gaussian_param_head`/`interm` 的参数当 “new params”；你加的新模块若不匹配命名，可能会被当成 pretrained、LR 被乘 `backbone_lr_multiplier`
  - **checkpoint 兼容**：Lightning ckpt 加载时若结构变更，需要处理 key mismatch（仓库里已有 `src/misc/weight_modify.py: checkpoint_filter_fn` 被引入但未在主流程显式调用；你若改结构可能需要接上）
  - **值域/shape 假设**：渲染输出 `DecoderOutput.color/depth/alpha` 被多种 loss 与 metric 复用，改输出格式会牵一堆

### 4.2 改损失 / 正则（Loss assembly, 权重调度, logging）
- **主要落点**
  - 新增 loss：在 `src/loss/` 新建 `loss_xxx.py`
  - 注册：`src/loss/__init__.py`（加 wrapper + 类到 `LOSSES`，并更新 union `LossCfgWrapper`）
  - 配置：`config/loss/xxx.yaml` + 在实验里 `override /loss: [.., xxx]` 并设置 `loss.xxx.*`
- **连锁影响**
  - `src/config.py` 的 union 解包依赖 wrapper 结构，**wrapper 写错会直接导致配置加载失败**
  - mask/valid 区域：很多 loss 使用 `batch["context"]["valid_mask"]` 或 distill 的 `conf_mask`，你新增 loss 要明确依赖哪个 mask（否则指标不可比）

### 4.3 改训练策略（optimizer/lr/amp/accum/EMA 等）
- **主要落点**
  - Lightning Trainer：`src/main.py`（`precision`, `accumulate_grad_batches`, strategy）
  - 优化器/调度器：`src/model/model_wrapper.py: configure_optimizers()`
  - 训练逻辑（skip batch、distill 等）：`ModelWrapper.training_step()`
- **连锁影响**
  - 混合精度：训练里 loss 计算显式 `autocast(enabled=False)`，你若引入新算子需注意 dtype
  - 多机多卡：目前用 `"ddp_find_unused_parameters_true"`；改网络后 unused params 可能变多/变少，影响性能与正确性

### 4.4 改数据 / 增强（dataset/collate/transform/sampler）
- **主要落点**
  - dataset 实现：`src/dataset/dataset_*.py`
  - view sampler：`src/dataset/view_sampler/*` + `config/dataset/view_sampler/*.yaml`
  - 混合采样/动态 batch：`src/dataset/data_sampler.py`（`MixedBatchSampler` 是关键）
  - 值域规范：`src/dataset/shims/normalize_shim.py`（把 `[0,1]→[-1,1]`）
- **连锁影响**
  - 你的采样策略一改，**train/val/test 可比性**会变（尤其是 num_context_views / target_views / gap）
  - batch 结构 key 变动会直接炸 `ModelWrapper`（大量硬编码 key）

### 4.5 改推理 / 后处理（inference pipeline / 导出 / 可视化）
- **主要落点**
  - feed-forward 推理：`src/model/model/anysplat.py: inference()`、以及 `inference.py`
  - NVS 推理评估：`src/eval_nvs.py`（包含 pose 推断与尺度对齐逻辑）
  - 导出：`src/model/ply_export.py`、或 `inference.py` 里调用的 `export_ply`
- **连锁影响**
  - pose 的尺度对齐：`eval_nvs.py` / `simple_trainer.py` 都有 `scale_factor` 对齐逻辑；你改 pose/尺度定义会影响全部可视化与指标

---

## 5) “Idea → Plan” 模板（给你复制给 Cursor 直接拆任务）

把下面模板粘贴给 Cursor，并把 `<...>` 填上你的改进点即可。

### 5.1 输入模板（你给 Cursor 的内容）
- **改进点描述**：`<一句话说清楚你要改什么：例如“把 voxelize 融合从 softmax(conf) 改成 attention 融合，并加一个 consistency loss”>`
- **动机/预期收益**：`<更稳定/更快/更准/更省显存/更好泛化…>`
- **不想动的部分**：`<例如“不要改数据读取，只改 encoder”>`
- **对比基线**：`<用哪个 experiment 配置做 baseline，例如 +experiment=dl3dv>`

### 5.2 Cursor 应输出的 Plan 结构（建议强制它按此格式）
#### A) 影响范围（模块/文件）
- **模型结构**：`src/model/encoder/anysplat.py`（具体函数/类：`EncoderAnySplat.forward`, `voxelizaton_with_fusion`, `gaussian_param_head`, `map_pdf_to_opacity`）
- **高斯参数化**：`src/model/encoder/common/gaussian_adapter.py`（若改参数/SH/scale/rot）
- **损失**：`src/loss/loss_<new>.py` + `src/loss/__init__.py` + `config/loss/<new>.yaml`
- **配置实验**：`config/experiment/<new_exp>.yaml`（或在原 exp 上 override）
- **训练策略**（如需）：`src/model/model_wrapper.py: configure_optimizers/training_step`

#### B) 最小可行改动（MVP）
- **MVP-1**：只改 `<一个函数>`，保证能 train 起来、loss 有数值、指标能跑
- **MVP-2**：加开关配置项（默认关闭），确保与 baseline 可对齐
- **MVP-3**：补最小日志（wandb 记录关键中间量/统计量）

#### C) 需要新增/改动的配置项
- `model.encoder.<your_flag>: bool`
- `model.encoder.<your_hparam>: ...`
- `loss.<your_loss>.weight: float`
- （如需）`optimizer.<...>` / `trainer.<...>`

#### D) 必要 sanity checks（比单元测试更实用）
- **shape check**：在 `training_step` 打印/断言关键 tensor：`context_image`, `pts_all`, `gaussians.means.shape`, `output.color.shape`
- **值域 check**：确认 encoder 输入是否为 `[0,1]`（与 `eval_nvs.py` 一致）
- **过拟合 1 个 batch**：看 loss 是否单调下降（避免 silent bug）
- **推理一致性**：`AnySplat.inference()` 与训练 forward 的 gaussians 统计量一致（均值/方差量级）

#### E) 对比实验矩阵（baseline vs ablation）
- **Baseline**：`python src/main.py +experiment=<baseline>`
- **Ablation-1（只开结构改动）**：`... model.encoder.<flag>=true`
- **Ablation-2（结构+loss）**：`... loss.<new>.weight=...`
- **Ablation-3（不同超参）**：`... model.encoder.<hparam>=a/b/c`
- **评估**：
  - NVS：`python src/eval_nvs.py ...`
  - CO3D Pose：`python src/eval_pose.py ...`

---

## 需要你补充 / 跑一次才能完全确认的点（我不会因此卡住结论，但你应优先确认）
- **validation 值域一致性**：训练/推理脚本都把 `[-1,1]→[0,1]` 再进 encoder，但 `validation_step()` 看起来没做；若你依赖 val 指标，请先跑一次确认是否存在值域 bug/或数据源不同导致没问题。
- **dataset 返回值域**：从 `normalize_shim` 与训练逻辑推断 dataset 很可能输出 `[0,1]`，但不同 dataset 实现（`dataset_*.py`）可能存在差异；建议用一次 batch dump 确认。

--- 

如果你把“你的改进点”用 3–5 句话发我（按 5.1 模板），我可以直接给出**精确到文件/函数级别**的“改造落地方案 + Cursor 可执行任务拆分（含配置 diff 与最小 sanity-check 列表）”。
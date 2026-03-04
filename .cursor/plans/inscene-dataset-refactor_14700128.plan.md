---
name: inscene-dataset-refactor
overview: 重命名当前仅针对 processed_infinigen 的 InsScene 实验／脚本，并为 processed_scannetpp_v2 设计与实现新的 manifest 生成与训练配置，为后续三子集联合的完整 InsScene-15K 训练打基础。
todos:
  - id: rename-inscene-infinigen-configs
    content: 梳理并重命名所有只覆盖 processed_infinigen 的实验/配置（instseg_insscene15k* → instseg_inscene_infinigen*），更新 VSCode 启动项与脚本示例。
    status: completed
  - id: unify-custom-schema-docs
    content: 在 DatasetCustom / DatasetCustomCfg 与 config/dataset/custom.yaml 中统一并中性化注释，明确它服务于 InsScene 各子集与其他多视角数据集的通用 manifest schema。
    status: completed
  - id: design-scannetpp-manifest
    content: 根据实际 scene_iphone_metadata.npz / scene_dslr_metadata.npz 的键与维度，敲定 ScanNet++ per-frame 与 per-scene manifest 的映射规则。
    status: completed
  - id: implement-scannetpp-manifest-script
    content: 新增 make_manifest_inscene_scannetpp_v2.py（或等价脚本），从 processed_scannetpp_v2_extracted 生成 manifest_scannetpp_v2.jsonl，与现有 DatasetCustom 完全兼容。
    status: completed
  - id: add-scannetpp-experiments
    content: 新建 instseg_inscene_scannetpp_v2.yaml（及多卡版本），配置数据根与 manifest 路径，并在 .vscode/launch.json 中加入对应训练入口。
    status: completed
  - id: design-re10k-support
    content: 设计（暂不实现）processed_re10k 的 manifest schema 与 DatasetCustom 对“无 depth”场景的支持策略。
    status: completed
  - id: plan-full-inscene15k-training
    content: 设计合并三个子集 manifest 的工具与最终 instseg_insscene15k.yaml，确保命名与“完整 InsScene-15K” 语义一致。
    status: completed
isProject: false
---

### 目标

- **命名纠正**：把当前仅覆盖 `processed_infinigen` 子集的配置/脚本从“insscene15k”中解耦，改成以 InsScene + 子集为单位的命名（例如 `instseg_inscene_infinigen`）。
- **多子集支持**：在保持 `DatasetCustom` / `DatasetCustomCfg` 统一消费 manifest 的前提下，为 `processed_scannetpp_v2`（优先）和后续的 `processed_re10k` 设计各自的 manifest 生成脚本与实验配置，并规划最终完整 InsScene-15K（三子集合并）训练方案。

### 现状简要

- **单场景 manifest 生成**：`[scripts/make_manifest_infinigen.py](scripts/make_manifest_infinigen.py)` 读取 `frames/Image|Depth|ObjectSegmentation/camera_x` 与 `camview/*.npz`，生成：
  - `{"scene_id": ..., "frames": [{"rgb_path", "depth_path", "instance_mask_path", "K_px", "c2w", "HW", "near", "far"}]}`（路径相对 `scene_dir`）。
- **InsScene-15K processed_infinigen 聚合**：`[scripts/extract_and_make_manifest_insscene.py](scripts/extract_and_make_manifest_insscene.py)` 从 `InsScene-15K/processed_infinigen/scene_*/xxx.zip` 解压到 `processed_infinigen_extracted/scene_xxx/subscene/`，对每个子场景调用 `make_scene_manifest`，将 frame 路径改为相对整个 `extract_dir`，写成多行 `.jsonl` manifest。
- **统一消费 manifest 的数据集**：
  - InstSeg 分支：`[src/instseg/dataset_custom.py](src/instseg/dataset_custom.py)` + `DatasetCustomCfg(name="custom", root, manifest_path, input_image_shape, near/far, ...)`。
  - 主分支：`[src/dataset/dataset_custom.py](src/dataset/dataset_custom.py)`，schema 与 InstSeg 版本基本一致，支持 `.jsonl` 和 `.json`，支持别名字段（`rgb_path/image_path/rgb` 等）。
- **实验与配置**：
  - 基础 InstSeg 模板：`[config/experiment/instseg_custom.yaml](config/experiment/instseg_custom.yaml)` 通过 `defaults: - /dataset@_group_.custom: custom` 使用 `[config/dataset/custom.yaml](config/dataset/custom.yaml)`，后者目前默认指向 Infinigen 示例场景。
  - 当前 InsScene 实验：`[config/experiment/instseg_insscene15k.yaml](config/experiment/instseg_insscene15k.yaml)` 与 `[config/experiment/instseg_insscene15k_mv.yaml](config/experiment/instseg_insscene15k_mv.yaml)` 实际上 **只覆盖 `processed_infinigen` 子集**，通过 `dataset.custom.root` 和 `dataset.custom.manifest_path` 指向 `processed_infinigen_extracted`。
  - VSCode 启动配置 `[.vscode/launch.json](.vscode/launch.json)` 中的 "Train instseg 8" 等项也仍使用 `+experiment=instseg_insscene15k_mv`。
- **其它子集原始结构（来自你的描述）**：
  - `processed_scannetpp_v2_extracted/processed_scannetpp_v2/<scene_id>/`：
    - `depth/`, `images/`, `refined_ins_ids/`, `scene_iphone_metadata.npz`（或 `scene_dslr_metadata.npz`，内含 `intrinsics.npy` / `trajectories.npy`）。
  - `processed_re10k`：只有 `cam`（多 view npz, 含 `intrinsic` 和 `pose`）、`rgb/` 与 `refined_ins_ids/`，**没有 depth**。

### 设计原则

- **统一消费层 schema**：尽量保持 `DatasetCustom` 的 frame-level schema 不变（`rgb_path`, `depth_path`[可选], `instance_mask_path`, `K_px`, `c2w`, `HW`, `near`, `far`），这样训练/推理逻辑无需为每个子集写一套 dataset class。
- **子集差异在生成层吸收**：
  - Infinigen：继续通过 `frames/`* + `camview` 转成统一 schema。
  - ScanNet++ v2：从 `scene_iphone_metadata.npz` / `scene_dslr_metadata.npz` 中解出 per-view `K` 与 `c2w`，与 `images/`、`depth/`、`refined_ins_ids/` 对齐后转成统一 schema。
  - RE10K：因 **缺少 depth**，在后续步骤中扩展 `DatasetCustom` 支持“无 depth”模式（仅加载 RGB+instance mask），并在 manifest 中允许缺失 `depth_path`；当前阶段专注实现 ScanNet++ v2 的有深度版本。
- **命名与配置解耦**：遵从你选择的策略——**让 `insscene15k` 只保留给“完整三子集”场景**，现有只用 `processed_infinigen` 的实验改名为 `instseg_inscene_infinigen`*。

### 计划步骤

#### 1. 重命名与整理现有 InsScene-processed_infinigen 实验

- **1.1 梳理所有 "insscene15k" 使用点**
  - 搜索代码库中 `insscene15k` 字符串（experiment 文件名、yaml 内容、脚本参数、文档、`.vscode/launch.json` 等）。
  - 确认哪些地方语义上“只代表 processed_infinigen 子集”，哪些将来应保留给“三子集联合”。
- **1.2 重命名实验 yaml 与内部字段**
  - 将 `[config/experiment/instseg_insscene15k.yaml](config/experiment/instseg_insscene15k.yaml)` 与 `[config/experiment/instseg_insscene15k_mv.yaml](config/experiment/instseg_insscene15k_mv.yaml)` 重命名为例如：
    - `config/experiment/instseg_inscene_infinigen.yaml`
    - `config/experiment/instseg_inscene_infinigen_mv.yaml`
  - 同步修改文件内部的：
    - `experiment_name` / `wandb.name` / `output_dir` 等中包含 `insscene15k` 的字段，改成 `inscene_infinigen` 风格。
    - 若有 `tags` / `notes` 提及“15k 完整数据集”，调整为“InsScene processed_infinigen 子集”。
- **1.3 更新 VSCode 配置与脚本调用**
  - 在 `[.vscode/launch.json](.vscode/launch.json)` 中：
    - `Train instseg 8` 等配置的 `+experiment=instseg_insscene15k_mv` 改为新名字（如 `+experiment=instseg_inscene_infinigen_mv`）。
  - 搜索脚本（如 `scripts/*.py`）中直接用到 `instseg_insscene15k`* 的命令示例，统一替换为新的实验名。
- **1.4 文案与帮助说明修正**
  - 在 `[scripts/extract_and_make_manifest_insscene.py](scripts/extract_and_make_manifest_insscene.py)` 中：
    - 将 help 文案从“processed_infinigen dir (scene_*/xxx.zip)”调整为更清晰的 InsScene 子集描写，如“InsScene-15K processed_infinigen 子集（scene_*/xxx.zip）”。
  - 在 `[config/dataset/custom.yaml](config/dataset/custom.yaml)` 中：
    - 把注释从 Infinigen-only 语义改为“通用多视角 manifest 数据集”的描述（例：`# Generic multi-view scene dataset (e.g. InsScene-Infinigen, ScanNet++, RE10K)`）。

#### 2. 抽象/整理通用 manifest schema（消费侧）

- **2.1 核心 schema 确认与注释统一**
  - 在 `DatasetCustom` 与 `DatasetCustomCfg`（`[src/dataset/dataset_custom.py](src/dataset/dataset_custom.py)` 与 `[src/instseg/dataset_custom.py](src/instseg/dataset_custom.py)`）中，将 class docstring 和字段注释统一为：
    - “自定义多视角数据集（InsScene 各子集、Infinigen、ScanNet++、RE10K 等）的通用 manifest schema”。
  - 明确文档化 frame 字段：
    - `rgb_path`, `depth_path` (可为 `null` 或缺失，以支持 RE10K)、`instance_mask_path`, `K_px`, `c2w`, `HW`, `near`, `far`。
- **2.2 预留无 depth 支持的接口设计（仅设计不实现）**
  - 在计划中记录：后续为 RE10K 添加可选 `depth_path` 的处理逻辑：
    - 如果某 frame 不含 `depth_path`，则：
      - dataset 不加载 depth，并在返回的 batch 中提供 `depth=None` 或全 0 + `valid_depth_mask=False`。
      - 所有依赖 depth 的 loss/可视化逻辑在 RE10K 实验中关闭或分支判断。
  - 当前阶段只在代码注释/设计文档中标明这一点，避免立刻改动训练主干。

#### 3. 为 processed_scannetpp_v2 设计 manifest 生成脚本

- **3.1 明确 ScanNet++ v2 目录与元数据结构**
  - 以你给的示例为基准：
    - `InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2/<scene_id>/`
      - `images/`：多 view RGB 图像（需要约定命名规则，例如 `frame_xxx.png`）。
      - `depth/`：与 images 对齐的深度图（npy 或 png）。
      - `refined_ins_ids/`：与 images 对齐的 instance id map（npy 或 png）。
      - `scene_iphone_metadata.npz`（或 `scene_dslr_metadata.npz`）：包含 `intrinsics.npy` & `trajectories.npy` 等数组（假设为 shape `[N,3,3]` / `[N,4,4]` 或 `[N,3,4]`）。
  - 在实现前先用一个小脚本（或交互式）检查实际 `npz` 内键名与维度，避免错误对齐（这一步你可自己快速跑，或等你允许我执行时我再写辅助检查脚本）。
- **3.2 设计统一的 per-scene manifest 构建逻辑**
  - 目标：产出与 Infinigen 相同的结构：
    - `{"scene_id": <scene_id>, "frames": [ ... ]}`，每个 frame：
      - `rgb_path`: 相对当前 `scene_dir` 的路径，例如 `images/XXXX.png`。
      - `depth_path`: `depth/XXXX.png` 或 `.npy`。
      - `instance_mask_path`: `refined_ins_ids/XXXX.npy` 或 `.png`。
      - `K_px`: 对应 view 的内参矩阵（像素坐标系）。
      - `c2w`: 对应 view 的 4×4 相机位姿矩阵。
      - `HW`: 原始分辨率 `[H, W]`（可从 metadata 或图像尺寸读取）。
      - `near/far`: 统一设定的扫描场景裁剪平面（可先沿用 0.01 / 100.0，后续如有需要再根据 ScanNet++ 范围调节）。
  - 视角对齐策略：
    - 约定一个 `parse_index` 规则（如从文件名中解析 frame index），或直接假设 `images`, `depth`, `refined_ins_ids`, metadata 中的 `intrinsics`/`trajectories` 都按同一顺序排列（使用枚举下标进行匹配）。
- **3.3 实现新的脚本（或系列脚本）**
  - 在 `scripts/` 下新增例如：
    - `make_manifest_inscene_scannetpp_v2.py`：
      - 输入：
        - `--data_root`：`InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2`。
        - `--manifest_out`：输出 `.jsonl` 路径（默认 `data_root/../manifest_scannetpp_v2.jsonl`）。
      - 逻辑：
        1. 枚举所有 `<scene_id>` 目录。
        2. 对每个 scene：
          - 解析 `scene_iphone_metadata.npz` / `scene_dslr_metadata.npz`，得到 per-view `K` 和 `c2w`。
          - 与 `images/`, `depth/`, `refined_ins_ids/` 对齐，构造 frames 列表。
          - 输出 `{"scene_id": scene_id, "frames": frames}` 一行到 manifest。
    - 也可以复用现有的 `make_scene_manifest` 思路，写一个新的 `make_scannet_scene_manifest(scene_dir, metadata_path, ...)`，与 Infinigen 版本结构对齐但专门适配 ScanNet++ 目录。

#### 4. 为 processed_scannetpp_v2 新增训练与调试配置

- **4.1 数据集 group 配置**
  - 在 `[config/dataset/custom.yaml](config/dataset/custom.yaml)` 基础上新增或重用 group，但通过 experiment override 指向 ScanNet++ manifest：
    - 在新的实验 yaml 中覆盖：
      - `dataset.custom.root: /.../InsScene-15K/processed_scannetpp_v2_extracted/processed_scannetpp_v2`
      - `dataset.custom.manifest_path: ../manifest_scannetpp_v2.jsonl`（或你在上一步决定的实际路径）。
    - 根据 ScanNet++ 的实际分辨率设置：
      - `original_image_shape`, `input_image_shape`（保证与 `scene_iphone_metadata` / 实际图像尺寸兼容）。
      - 如 ScanNet++ 深度范围更小，可适当调整 `near`/`far`，但初期可以沿用 Infinigen/InsScene 设置先跑通。
- **4.2 InstSeg 实验定义**
  - 新增例如：
    - `[config/experiment/instseg_inscene_scannetpp_v2.yaml](config/experiment/instseg_inscene_scannetpp_v2.yaml)`：
      - `defaults`: 继承 `instseg_custom`，再覆盖 `dataset.custom.`* 到 ScanNet++ 路径与 manifest。
      - 训练参数：batch size / num_steps / lr 等可先复制 `instseg_inscene_infinigen.yaml`，后续根据显存和数据规模微调。
  - 如需多卡版本，再复制出 `instseg_inscene_scannetpp_v2_mv.yaml`，比照 `instseg_inscene_infinigen_mv.yaml` 配置 DDP 与 `CUDA_VISIBLE_DEVICES`。
- **4.3 VSCode & 调试脚本支持**
  - 在 `[.vscode/launch.json](.vscode/launch.json)` 中增加新的启动项：
    - 例如 "Train instseg ScanNet++"，`+experiment=instseg_inscene_scannetpp_v2`。
  - 若有需要，用 `overfit_one_scene.py` 或 `scripts/debug_instseg_ckpt*.py` 增加 ScanNet++专用示例命令行（修改 `--run_dir` 指向 ScanNet++ 的训练输出）。

#### 5. 规划后续 processed_re10k 与完整 InsScene-15K 训练（设计阶段）

- **5.1 processed_re10k manifest 设计草案**
  - 基于你提供的信息：只有 `cam` (npz with `intrinsic` & `pose`), `rgb`, `refined_ins_ids`，无 depth：
    - 设计 manifest 时：
      - `rgb_path` 与 `instance_mask_path` 正常填入。
      - `depth_path` 字段可以：
        - 方案 A：省略（从而触发后续 `DatasetCustom` 的“可选深度”逻辑）。
        - 方案 B：填入一个虚拟路径并在 loader 中专门处理（不建议）。
      - `K_px` 与 `c2w` 从 `cam` npz 读取；`HW` 从图像分辨率读取；`near/far` 按 RE10K 相机范围设置一个合理区间。
  - 在当前计划中先不实现，而是在 ScanNet++ 跑通后，单独开一个小步骤扩展 `DatasetCustom`，并新增 `instseg_inscene_re10k.yaml`。
- **5.2 完整 InsScene-15K（三级合并）训练方案草案**
  - Manifest 合并层：
    - 最简单的方式：
      - 拥有三个 `.jsonl`：`manifest_infinigen.jsonl`, `manifest_scannetpp_v2.jsonl`, `manifest_re10k.jsonl`。
      - 新建脚本 `merge_manifests_inscene15k.py`，简单逐行读取三个文件，把所有 scene 依次写入一个大的 `manifest_inscene15k.jsonl`，`scene_id` 可以带前缀例如 `inf_...`, `spp_...`, `re10k_...` 以避免冲突。
  - 实验配置：
    - 新建真正意义上的 `[config/experiment/instseg_insscene15k.yaml](config/experiment/instseg_insscene15k.yaml)`：
      - `dataset.custom.root`：指向三个子集的“公共根”（或用软链接统一到一个伪根下）。
      - `dataset.custom.manifest_path`：指向 `manifest_inscene15k.jsonl`。
    - 这样“insscene15k” 这个名字就与“完整三子集联合训练”精确对齐，不再与单一子集混淆。

### 数据流示意（子集 → Manifest → 训练）

```mermaid
flowchart LR
  subgraph infinigen [InsScene-Infinigen 子集]
    infinigenZips["processed_infinigen scene_*/xxx.zip"] --> infinigenExtract[extract_and_make_manifest_insscene.py]
    infinigenExtract --> infManifest["manifest_infinigen.jsonl"]
  end

  subgraph scannetpp [InsScene-ScanNet++ v2 子集]
    scannetDir["processed_scannetpp_v2_extracted/processed_scannetpp_v2"] --> scannetScript[make_manifest_inscene_scannetpp_v2.py]
    scannetScript --> sppManifest["manifest_scannetpp_v2.jsonl"]
  end

  subgraph re10k [InsScene-RE10K 子集]
    re10kDir["processed_re10k"] --> re10kScript[make_manifest_inscene_re10k.py (规划阶段)]
    re10kScript --> re10kManifest["manifest_re10k.jsonl"]
  end

  infManifest & sppManifest & re10kManifest --> merged["merge_manifests_inscene15k.py"] --> fullManifest["manifest_inscene15k.jsonl"]

  fullManifest --> datasetCfg["DatasetCustomCfg (custom)"] --> datasetCustom["DatasetCustom / InstSegDataModule"] --> training["instseg_insscene15k 实验"]
```



### TODO 列表

- **rename-inscene-infinigen-configs**: 梳理并重命名所有只覆盖 `processed_infinigen` 的实验/配置（`instseg_insscene15k`* → `instseg_inscene_infinigen`*），更新 VSCode 启动项与脚本示例。
- **unify-custom-schema-docs**: 在 `DatasetCustom` / `DatasetCustomCfg` 与 `config/dataset/custom.yaml` 中统一并中性化注释，明确它服务于 InsScene 各子集与其他多视角数据集的通用 manifest schema。
- **design-scannetpp-manifest**: 根据实际 `scene_iphone_metadata.npz` / `scene_dslr_metadata.npz` 的键与维度，敲定 ScanNet++ per-frame 与 per-scene manifest 的映射规则。
- **implement-scannetpp-manifest-script**: 新增 `make_manifest_inscene_scannetpp_v2.py`（或等价脚本），从 `processed_scannetpp_v2_extracted` 生成 `manifest_scannetpp_v2.jsonl`，与现有 `DatasetCustom` 完全兼容。
- **add-scannetpp-experiments**: 新建 `instseg_inscene_scannetpp_v2.yaml`（及多卡版本），配置数据根与 manifest 路径，并在 `.vscode/launch.json` 中加入对应训练入口。
- **design-re10k-support**: 设计（暂不实现）`processed_re10k` 的 manifest schema 与 `DatasetCustom` 对“无 depth”场景的支持策略。
- **plan-full-inscene15k-training**: 设计合并三个子集 manifest 的工具与最终 `instseg_insscene15k.yaml`，确保命名与“完整 InsScene-15K” 语义一致。


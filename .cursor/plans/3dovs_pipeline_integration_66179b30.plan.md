---
name: 3dovs Pipeline Integration
overview: 将 3dovs 数据集适配到训练管线，包含数据预处理脚本、DatasetCustom 的 depth 可选化改造、以及 validation_step 中的物理预测可视化输出，使训练过程中每个 val step 都能看到预测结果。
todos:
  - id: prepare-script
    content: "新建 scripts/prepare_3dovs.py: 解析 COLMAP + phys_params.json, 生成 manifest.jsonl + physics_labels.json"
    status: completed
  - id: depth-optional
    content: "修改 DatasetCustom: depth 变为可选, 缺失时生成全 True valid_mask"
    status: completed
  - id: val-phys-output
    content: "修改 IGGTWrapper.validation_step: 输出 per-instance 物理预测结果到日志"
    status: completed
  - id: config-3dovs
    content: 调整实验配置指向 3dovs/bench 数据, 可跑通训练
    status: completed
isProject: false
---

# 3dovs 数据管线集成

## 数据现状

- 36 张图片 (1008x756 JPG), 36 个 id_map (504x336 int32 NPY, ID 0-7)
- COLMAP SIMPLE_RADIAL 相机 (cameras.bin + images.bin)
- `phys_params.json`: 8 个实例，标签为 `static` / `dynamic_rigid` / `dynamic_soft` / `out_of_range`
- 无深度数据

## 1. 数据预处理脚本

**新建**: `scripts/prepare_3dovs.py`

功能：

- 解析 COLMAP `cameras.bin` → 提取 SIMPLE_RADIAL 内参，转为 3x3 K 矩阵（去掉径向畸变参数）
- 解析 COLMAP `images.bin` → 提取每张图的 quaternion + translation，转为 4x4 c2w 矩阵
- 从 `phys_params.json` 提取 `id` → `response.behavior_template`，转换标签名称：
  - `static` → `static`, `dynamic_rigid` → `rigid`, `dynamic_soft` → `soft`, `out_of_range` → `unknown`
- 输出到场景目录下:
  - `physics_labels.json`: `{"0": "rigid", "1": "soft", ...}`
  - `manifest.jsonl`: 一行一个 scene，包含所有 frame 的 rgb_path / instance_mask_path / K / c2w / HW

manifest 格式示例：

```json
{"scene_id": "bench", "physics_labels_path": "physics_labels.json", "frames": [
  {"rgb_path": "images/00.jpg", "instance_mask_path": "id_maps/00.npy",
   "K_px": [[779.26, 0, 504], [0, 779.26, 378], [0, 0, 1]],
   "c2w": [[...4x4...]], "HW": [756, 1008]},
  ...
]}
```

**注意**: id_map 分辨率 (504x336) 与图片 (1008x756) 不同，但 `crop_shim` 会各自 rescale 到目标分辨率，所以可以直接用。manifest 中 `HW` 记录的是图片的原始分辨率。

## 2. DatasetCustom depth 可选化

**修改**: [src/dataset/dataset_custom.py](src/dataset/dataset_custom.py)

在 `load_stack` 中，当 frame 没有 `depth_path` / `depth` 字段时：

- 跳过 depth 加载，生成与 instance_mask 同尺寸的 **全 True valid_mask**（所有像素视为有效）
- `depth` 填为全零 tensor（占位，不影响训练——physics 实验不用 depth loss）

## 3. Validation step 中输出物理预测

**修改**: [src/model/iggt_wrapper.py](src/model/iggt_wrapper.py) 的 `validation_step`

在 validation 时：

1. 取 `encoder_output.physics_feat_map` [B, V, C, H, W]
2. 用 batch 中的 GT `instance_mask` 做 per-instance masked pooling
3. 通过 `LossPhys` 的 classifier（或单独加载）得到 per-instance logits
4. 对每个 instance 输出: `{id: N, predicted: "rigid", gt: "soft", confidence: 0.85}`
5. 用 `logger.info` 打印到控制台，同时 log 到 tensorboard

这样每隔 `val_check_interval` 步就能在日志里直接看到每个 instance 的预测结果。

## 4. 实验配置调整

**修改**: [config/experiment/phys_iggt_custom.yaml](config/experiment/phys_iggt_custom.yaml)

- `dataset.custom.root` 指向 `3dovs/bench`
- `dataset.custom.manifest_path` 指向生成的 `manifest.jsonl`
- `original_image_shape: [756, 1008]`
- 由于只有 1 个场景，可设 `overfit_to_scene` 或减小 val interval 方便调试


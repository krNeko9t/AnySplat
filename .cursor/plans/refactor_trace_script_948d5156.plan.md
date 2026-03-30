---
name: Refactor trace script
overview: 将 trace_instance_to_gaussians.py 重构为可插拔的 feature source 架构，支持 GT ID map（Random Embedding 编解码）、通用 map 输入、以及未来的 physics feature 等多种 trace 输入源。
todos:
  - id: add-idmap-codec
    content: 新增 IDMapCodec 类（Random Embedding 编解码）和 load_gt_idmaps 加载函数
    status: completed
  - id: add-prepare-functions
    content: 将 AnySplat/IGGT/precomputed 的特征准备逻辑抽取为 prepare_*_features() 函数，新增 prepare_gt_idmap_features()
    status: completed
  - id: refactor-main-trace-loop
    content: 重构 main() 中 trace loop：统一 feature source dispatch + 统一 trace 循环 + source-specific 后解码
    status: completed
  - id: add-cli-args
    content: 新增 --feature_source, --idmap_dir, --id_embed_dim, --id_embed_seed 命令行参数，保持向后兼容
    status: completed
  - id: adapt-postprocess
    content: 后处理适配：新增 gt_color 模式用 ID 直接着色，gt_idmap 模式跳过不必要的聚类，输出文件名带 source 标签
    status: completed
  - id: test-backward-compat
    content: 验证不指定 --feature_source 时的自动推断逻辑，确保现有用法完全兼容
    status: completed
isProject: false
---

# 重构 trace 脚本：可插拔 Feature Source 架构

## 现状分析

当前 `[scripts/trace_instance_to_gaussians.py](scripts/trace_instance_to_gaussians.py)` 将 **特征生成**、**trace 核心**、**后处理** 耦合在一个 main() 中。三种输入模式（AnySplat/IGGT 在线推理、预计算 .pt/.npy）的逻辑交织在 trace loop 里，添加新 feature source 需要大量 if/else。

## 核心设计

引入 **Feature Provider** 概念，将 `main()` 中的 trace loop 解耦为三阶段：

```
[Feature Provider] --> per-view (D, H, W) tensor
                          |
                    [Trace Core] --> (N, D) accumulated features
                          |
                    [Post-decode] --> optional ID recovery / label assignment
```

### Feature Source 类型（通过 `--feature_source` 参数选择）

- `anysplat` — 现有 Mode A/AnySplat，448x448 crop-aligned trace
- `iggt` — 现有 Mode A/IGGT，resize-aligned trace
- `precomputed` — 现有 Mode B，加载 .pt/.npy 连续特征（也作为通用 map 入口）
- `gt_idmap` — **新增**：加载整数 ID map，Random Embedding 编码后 trace，trace 后解码回 ID

### 关键约束

- `TRACE_CHANNELS = 20`（CUDA kernel 编译常量），feat_dim 不超过 20
- GT ID map 使用 Random Embedding：每个唯一 ID 映射到随机 D 维向量（默认 D=16），trace 后用最近邻恢复 ID
- 相机对齐：`gt_idmap` 和 `precomputed` 模式使用原始相机分辨率，将 map resize 到匹配

## 文件改动

仅修改 `[scripts/trace_instance_to_gaussians.py](scripts/trace_instance_to_gaussians.py)`，不拆分文件（脚本性质，保持单文件可用性）。

### 1. 新增 GT ID Map 加载与编解码模块

在 "Mode B" 区域后新增：

```python
class IDMapCodec:
    """Encode discrete integer IDs to random embeddings and decode back."""
    
    def __init__(self, embed_dim=16, seed=42):
        self.embed_dim = embed_dim
        self.seed = seed
        self.id_to_idx = {}      # unique_id -> table index
        self.embedding_table = None  # (num_ids, embed_dim) tensor
    
    def fit(self, id_maps):
        """Scan all ID maps to discover unique IDs, build embedding table."""
        all_ids = set()
        for m in id_maps:
            all_ids.update(m.unique().tolist())
        all_ids.discard(0)  # 0 = background/ignore
        sorted_ids = sorted(all_ids)
        self.id_to_idx = {uid: i+1 for i, uid in enumerate(sorted_ids)}
        # index 0 reserved for background
        n = len(sorted_ids) + 1
        g = torch.Generator().manual_seed(self.seed)
        self.embedding_table = F.normalize(
            torch.randn(n, self.embed_dim, generator=g), p=2, dim=-1
        )
        self.embedding_table[0] = 0  # background -> zero vector
    
    def encode(self, id_map):
        """(H, W) int -> (embed_dim, H, W) float embedding map."""
        H, W = id_map.shape
        idx_map = torch.zeros(H, W, dtype=torch.long)
        for uid, tidx in self.id_to_idx.items():
            idx_map[id_map == uid] = tidx
        return self.embedding_table[idx_map].permute(2, 0, 1).contiguous()
    
    def decode(self, feat, valid_mask):
        """(G, D) traced features -> (G,) integer IDs via nearest neighbor."""
        table = self.embedding_table.to(feat.device)
        sim = feat @ table.T  # (G, num_ids)
        pred_idx = sim.argmax(dim=1)
        idx_to_id = {0: 0}
        idx_to_id.update({v: k for k, v in self.id_to_idx.items()})
        ids = torch.zeros(feat.shape[0], dtype=torch.int64, device=feat.device)
        for tidx, uid in idx_to_id.items():
            ids[pred_idx == tidx] = uid
        ids[~valid_mask] = 0
        return ids
```

GT ID map 加载函数（复用项目中 `_load_instance_mask` 的逻辑）：

```python
def load_gt_idmaps(idmap_dir, image_names):
    """Load per-view integer ID maps from idmap_dir/{name}.npy or image files."""
    ...
```

### 2. 重构 main() 的 trace loop

将当前 main() 中 ~150 行的 mode_a/mode_b 交错逻辑重构为：

```python
# --- Prepare feature source ---
if feature_source == "anysplat":
    feat_maps, trace_cams, feat_dim = prepare_anysplat_features(...)
elif feature_source == "iggt":
    feat_maps, trace_cams, feat_dim = prepare_iggt_features(...)
elif feature_source == "precomputed":
    feat_maps, trace_cams, feat_dim = prepare_precomputed_features(...)
elif feature_source == "gt_idmap":
    feat_maps, trace_cams, feat_dim, id_codec = prepare_gt_idmap_features(...)

# --- Unified trace loop (unchanged core) ---
for idx, cam in enumerate(tqdm(cam_list)):
    trace_cam = trace_cams[idx]
    feat_2d = feat_maps[idx]
    # ... resize if needed, zero-pad, trace_single_view, accumulate ...

# --- Post-decode (source-specific) ---
if feature_source == "gt_idmap":
    gaussian_ids = id_codec.decode(final_feat, valid_mask)
    # save as .npy, also generate colored PLY
```

### 3. 新增命令行参数

```
--feature_source {anysplat,iggt,precomputed,gt_idmap}  # 替代隐式 mode_a/mode_b 判断
--idmap_dir DIR          # gt_idmap 模式：ID map 文件目录
--id_embed_dim INT       # gt_idmap 模式：embedding 维度（默认 16）
--id_embed_seed INT      # gt_idmap 模式：随机种子
```

保留所有现有参数的兼容性。若不指定 `--feature_source`，则按现有逻辑自动推断（`--run_dir` 存在 -> anysplat/iggt，`--feat_dir` 存在 -> precomputed）。

### 4. 后处理适配

- 对 `gt_idmap` 模式，`--postprocess` 中的 clustering 步骤变为可选（因为 ID 已经是语义标签，不需要聚类）
- 新增 `--postprocess gt_color`：直接用 ID 着色 PLY（跳过 PCA/聚类）
- PCA 可视化对所有 source 都保持可用

### 5. 保存格式扩展

输出文件名从 `gaussian_instance_feat.pt/npy` 改为带 source 标签：

- `gaussian_{source}_feat.pt` / `.npy`（通用）
- `gt_idmap` 模式额外保存 `gaussian_gt_ids.npy`（解码后的整数 ID）

## 使用示例

```bash
# GT ID Map trace
python scripts/trace_instance_to_gaussians.py \
    --source_path scene/ --ply_path scene/point_cloud.ply \
    --feature_source gt_idmap --idmap_dir scene/id_maps/ \
    --id_embed_dim 16 --max_views 20 \
    --postprocess gt_color,pca

# 通用 map（如 physics features 预计算后保存的 .npy）
python scripts/trace_instance_to_gaussians.py \
    --source_path scene/ --ply_path scene/point_cloud.ply \
    --feature_source precomputed --feat_dir phys_feats/ --feat_dim 8 \
    --max_views 20 --postprocess pca

# 现有 AnySplat/IGGT 用法完全不变
python scripts/trace_instance_to_gaussians.py \
    --source_path scene/ --ply_path scene/point_cloud.ply \
    --run_dir output/exp/ --max_views 5
```


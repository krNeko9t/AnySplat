重构完成。以下是改动总结：

---

## 改动总结

### 新增代码

1. **`IDMapCodec` 类**（第 592-653 行）
   - `fit(id_maps)`: 扫描所有 ID map 收集唯一 ID，为每个 ID 生成 L2-normalized 随机嵌入向量（背景 ID=0 → 零向量）
   - `encode(id_map)`: `(H, W) int → (D, H, W) float` 嵌入特征图
   - `decode(feat, valid_mask)`: `(G, D) → (G,) int` 通过余弦最近邻恢复 ID

2. **`load_gt_idmaps()`**（第 656-685 行）：加载 `.npy`、`.png` 或其他图像格式的 GT ID map

3. **四个 `prepare_*_features()` 函数**（第 691-814 行）：
   - `prepare_anysplat_features` / `prepare_iggt_features` — 封装了原来 main() 里 mode_a 的全部逻辑
   - `prepare_precomputed_features` — 封装了原 mode_b
   - `prepare_gt_idmap_features` — **新增**，加载 ID map → 编码为嵌入 → 生成 masks

4. **`_resolve_feature_source()` + `_validate_source_args()`**（第 1219-1263 行）：自动推断或验证 feature source

### 修改的代码

5. **`run_postprocess()`**：新增 `gaussian_ids` 参数和 `gt_color` 后处理模式（直接用 GT ID 着色）

6. **`render_colored_views()`**：扫描 PLY 时新增 `gt_color/colored_gaussians.ply`

7. **`main()`** — 重构为统一的三阶段流水线：
   - **Prepare**: 根据 `--feature_source`（或自动推断）调用对应的 prepare 函数
   - **Trace**: 统一的 trace 循环，支持可选的 per-view masks
   - **Post-decode**: `gt_idmap` 模式自动解码嵌入→整数 ID

### 新增命令行参数

| 参数 | 说明 |
|------|------|
| `--feature_source` | `{anysplat, iggt, precomputed, gt_idmap}` — 不指定则按旧参数自动推断 |
| `--idmap_dir` | GT ID map 文件目录 |
| `--id_embed_dim` | 随机嵌入维度（默认 16） |
| `--id_embed_seed` | 嵌入表随机种子（默认 42） |

### 向后兼容

- 所有旧参数（`--run_dir`, `--feat_dir`, `--model_type iggt` 等）完全保留
- 不指定 `--feature_source` 时，自动按旧逻辑推断源类型
- 自动推断模式下输出文件名保持 `gaussian_instance_feat.pt/npy`
零 lint 错误。下面总结一下本次重构完成的所有内容。

---

## 重构完成总结

### 新增文件

| 文件 | 用途 |
|------|------|
| `src/coord/__init__.py` | 公共 API 导出 |
| `src/coord/conventions.py` | 枚举（CameraConvention, ExtrinsicType, IntrinsicConvention）+ 轴分组 + ASCII 图 |
| `src/coord/camera_pose.py` | `CameraPose` 类 —— 唯一的坐标系转换入口 `.to()` |
| `src/coord/conversions.py` | 内部实现：轴查表 `convert_axes`、`se3_inv`、`qvec_to_rotmat` |
| `src/coord/intrinsics.py` | 内参像素中心转换（COLMAP ↔ OpenCV） |
| `src/coord/ply_io.py` | 统一 PLY 导入/导出，PLY header 写入 `coordinate_convention=` |
| `src/coord/_logging.py` | `log_coordinate_op` 和 `assert_convention` 工具 |
| `tests/test_coord.py` | 49 个单元测试（全部通过） |
| `.cursor/rules/coordinate-conventions.mdc` | AI 编码规则文件 |

### 修改文件（去重 + 标注）

| 文件 | 改动 |
|------|------|
| `src/model/ply_export.py` | 改为 `export_gaussian_ply` 的 thin wrapper |
| `src/model/encoder/vggt/utils/geometry.py` | `closed_form_inverse_se3` 改为委托 `se3_inv` |
| `src/utils/geometry.py` | 内参函数替换为 re-export |
| `src/geometry/ptc_geometry.py` | 同上 |
| `src/dataset/shims/geometry_shim.py` | 同上 |
| `src/dataset/dataset_scannetpp.py` | `blender2opencv_c2w` 改为使用 `CameraPose.to()`；添加类文档 |
| `src/dataset/dataset_dl3dv.py` | 同上 |
| `src/dataset/dataset_co3d.py` | 添加坐标系约定文档 |
| `src/dataset/dataset_custom.py` | 补充完整的坐标系要求文档（像素中心、OpenCV 约定） |
| `src/utils/render.py` | 散落的 `diag(1,-1,-1,1)` 添加注释说明 |
| `src/utils/viz.py` | `OPENGL` 常量添加注释关联到 `src.coord.AXIS_FLIP` |

### 核心设计

- **`CameraPose.to()` 是唯一的转换入口**，不再有 N*N 命名函数
- **`se3_inv`** 短名字替代 `closed_form_inverse_se3`，旧名保留为兼容别名
- PLY 文件 header 自动写入/读取坐标系标记，运行时可追踪
- 所有数据集类的 docstring 标注了原始数据约定 → 输出约定的映射
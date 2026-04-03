---
name: Coordinate System Refactor
overview: 创建统一的坐标系基础模块 `src/coord/`，定义坐标系约定枚举、带元数据的 CameraPose 类型、标准转换函数，并在项目的数据加载/PLY导出/渲染等关键边界处集成，消除重复代码并添加运行时可追踪性。范围仅限 `src/` 和 `scripts/`，不涉及根目录 `IGGT/`。
todos:
  - id: create-coord-module
    content: "创建 src/coord/ 模块：conventions.py (枚举+轴分组+ASCII图)、camera_pose.py (CameraPose 类，唯一转换入口 .to())、conversions.py (内部实现：轴查表+se3_inv+qvec_to_rotmat)、intrinsics.py (内参转换)、_logging.py (日志工具)、__init__.py (公共API)"
    status: pending
  - id: unify-ply-io
    content: 创建 src/coord/ply_io.py：统一 PLY 读写，合并 ply_export.py / simple_trainer.py / trace 脚本中的 3 套实现，PLY header 中写入坐标系 comment
    status: pending
  - id: dedup-blender2opencv
    content: "用 CameraPose(mat, BLENDER, C2W).to(OPENCV).matrix 替换 dataset_scannetpp.py、dataset_dl3dv.py 中的 2 处 blender2opencv_c2w 重复（IGGT 不管）"
    status: pending
  - id: dedup-inverse-se3
    content: "统一 se3_inv 到 src/coord/conversions.py；vggt/utils/geometry.py 的 closed_form_inverse_se3 改为 re-export 别名（IGGT 不管）"
    status: pending
  - id: dedup-intrinsics
    content: 统一 colmap_to_opencv_intrinsics / opencv_to_colmap_intrinsics 到 src/coord/intrinsics.py，替换 src/utils/geometry.py、src/geometry/ptc_geometry.py、src/dataset/shims/geometry_shim.py 3 处重复
    status: pending
  - id: dataset-convention-map
    content: 为每个数据集 loader 在加载边界处标注原始约定 -> 输出约定的映射，在 __getitem__ 添加 log_coordinate_op 日志；补充 dataset_custom.py docstring 中缺失的像素中心/世界轴说明
    status: pending
  - id: add-boundary-logging
    content: 在 PLY export、rendering 入口 (simple_trainer.py rasterize_splats、cuda_splatting.py)、scripts/trace 等关键边界处添加坐标系日志
    status: pending
  - id: create-cursor-rule
    content: 创建 .cursor/rules/coordinate-conventions.mdc，写入项目坐标系约定规则 + 各数据集约定速查表，供 AI 参考
    status: pending
  - id: add-tests
    content: 为坐标系转换写 round-trip 单元测试，确保 A -> B -> A == identity
    status: pending
isProject: false
---

# 坐标系统一重构计划

**范围**：仅 `src/` 和 `scripts/`。根目录 `IGGT/` 文件夹不在此次重构范围内。

## 现状分析

项目中与坐标系相关的代码分布在 15+ 个文件（`src/` 内），存在以下核心问题：

- **重复实现**：`blender2opencv_c2w` 在 [dataset_scannetpp.py](src/dataset/dataset_scannetpp.py) 和 [dataset_dl3dv.py](src/dataset/dataset_dl3dv.py) 各写一遍；`closed_form_inverse_se3` 在 [vggt/utils/geometry.py](src/model/encoder/vggt/utils/geometry.py) 是权威实现，被多处直接调用；`colmap_to_opencv_intrinsics` 在 3 处重复
- **约定不统一**：[cam_utils.py](src/misc/cam_utils.py) `update_pose` 注释说 extrinsics 是 c2w；[pose_enc.py](src/model/encoder/vggt/utils/pose_enc.py) 说是 "cam from world"（w2c）；[projection.py](src/geometry/projection.py) `transform_world2cam` 假定 extrinsics 是 c2w 然后内部取逆
- **隐式坐标翻转**：`diag(1,-1,-1,1)` 散落在 [render.py](src/utils/render.py)、[viz.py](src/utils/viz.py) 等处，没有统一常量名
- **PLY 导出 3 套**：[ply_export.py](src/model/ply_export.py) 用 `scales.log()`；[simple_trainer.py](src/post_opt/simple_trainer.py) 直接写 scales；[trace_instance_to_gaussians.py](scripts/trace_instance_to_gaussians.py) 读时假设是 log
- **无运行时追踪**：所有坐标假设只能靠读代码推断
- **数据集约定隐式**：各 loader 将不同源格式转为 "OpenCV c2w"，但转换链无日志、无类型标记

## 坐标系约定速查

- **OpenCV**：相机轴 X-right, Y-down, Z-forward；内参像素中心 (0, 0)
- **COLMAP**：相机轴同 OpenCV；内参像素中心 (0.5, 0.5)；外参默认存 w2c
- **OpenGL**：相机轴 X-right, Y-up, Z-backward；用于渲染投影矩阵
- **Blender**：相机空间同 OpenGL；世界坐标 Z-up, Y-forward；`transform_matrix` 为 c2w
- **NerfStudio**：相机空间同 OpenGL；世界坐标 Z-up

OpenCV/COLMAP 与 OpenGL/Blender/NerfStudio 之间的相机坐标转换矩阵：右乘 `diag(1, -1, -1, 1)`。

## 各数据集坐标流

当前项目统一的内部约定是 **OpenCV 相机系 + c2w 外参 + 归一化内参**。各 loader 的转换路径：

- **CO3D** ([dataset_co3d.py](src/dataset/dataset_co3d.py))：npz `camera_pose` 直接作为 c2w（已是 OpenCV 系）；K 除以 W,H 归一化
- **DL3DV** ([dataset_dl3dv.py](src/dataset/dataset_dl3dv.py))：`transforms.json` 的 `transform_matrix` 为 **Blender c2w** -> 右乘 `diag(1,-1,-1,1)` -> **OpenCV c2w**；K 从 `fl_x/fl_y/cx/cy` 构造并归一化
- **ScanNet++** ([dataset_scannetpp.py](src/dataset/dataset_scannetpp.py))：npz `trajectories` 直接作为 c2w（预处理已转为 OpenCV 系）；K 归一化
- **Custom** ([dataset_custom.py](src/dataset/dataset_custom.py))：manifest 要求用户提供 **c2w + 像素 K**，不做坐标系转换；**缺少对像素中心约定和世界轴的文档说明**
- **COLMAP (post_opt)** ([post_opt/datasets/colmap.py](src/post_opt/datasets/colmap.py))：从 pycolmap 读 w2c -> `inv()` 得 c2w；可选 `normalize.py` 对齐到 z-up

**关键风险点**：

- Custom 数据集不验证用户提供的 c2w 是否真的是 OpenCV 系
- DL3DV 和 ScanNet++ 都有 `blender2opencv_c2w` 的独立复制
- `colmap_to_opencv_intrinsics` 存在但各 loader **均未自动调用**，若原始 K 来自 COLMAP 会差 0.5 像素

## 设计方案

### 1. 新模块 `src/coord/`

```
src/coord/
  __init__.py           # 公共 API 导出
  conventions.py        # 枚举 + 静态转换矩阵 + ASCII 轴方向图
  camera_pose.py        # CameraPose 带元数据包装
  intrinsics.py         # 内参转换 (colmap <-> opencv 像素中心) + IntrinsicConvention 枚举
  conversions.py        # 所有坐标系转换函数 + closed_form_inverse_se3
  ply_io.py             # 统一 PLY 读写，带坐标系标记
  _logging.py           # 运行时坐标日志/断言工具
```

### 2. `conventions.py` -- 枚举与常量

定义 `CameraConvention` 枚举（OPENCV, OPENGL, COLMAP, BLENDER, NERFSTUDIO）和 `ExtrinsicType` 枚举（C2W, W2C）。顶部用 ASCII 图画出各坐标系的轴方向。提供静态 4x4 转换矩阵常量：

```python
from enum import Enum
import torch

class CameraConvention(str, Enum):
    OPENCV = "opencv"       # X-right, Y-down, Z-forward
    COLMAP = "colmap"       # 同 opencv 相机轴; 内参像素中心 (0.5, 0.5)
    OPENGL = "opengl"       # X-right, Y-up, Z-backward
    BLENDER = "blender"     # 相机空间同 opengl; 世界 Z-up
    NERFSTUDIO = "nerfstudio"  # 相机空间同 opengl; 世界 Z-up

class ExtrinsicType(str, Enum):
    C2W = "c2w"
    W2C = "w2c"

class IntrinsicConvention(str, Enum):
    OPENCV = "opencv"       # 像素中心 (0, 0)
    COLMAP = "colmap"       # 像素中心 (0.5, 0.5)

OPENCV_TO_OPENGL = torch.diag(torch.tensor([1., -1., -1., 1.]))
OPENGL_TO_OPENCV = OPENCV_TO_OPENGL  # self-inverse
```

### 3. `camera_pose.py` -- CameraPose：唯一的转换入口

设计哲学：**不写 N*N 个 `A_to_B()` 函数，所有转换只通过 `CameraPose.to()` 一个方法完成**。坐标系约定作为元数据跟着数据走，而不是靠开发者/AI 记住。

```python
class CameraPose:
    """带坐标系元数据的相机位姿。所有约定转换的唯一入口。"""
    matrix: Tensor                  # (*, 4, 4) SE3 矩阵
    convention: CameraConvention    # 哪种相机轴约定
    extrinsic_type: ExtrinsicType   # c2w 还是 w2c

    # ---- 唯一的转换方法 ----
    def to(self, convention=None, extrinsic_type=None) -> "CameraPose":
        """转到任意目标约定。不传的参数保持不变。
        
        用法：
            pose.to(OPENCV)                    # 只改相机轴
            pose.to(extrinsic_type=W2C)        # 只翻转 c2w/w2c
            pose.to(OPENCV, W2C)               # 同时改两者
        """
        ...
    
    # ---- 常用快捷方法 ----
    @property
    def c2w(self) -> Tensor:
        """返回 c2w 矩阵（当前约定下）。若已是 c2w 直接返回，否则取逆。"""
    
    @property
    def w2c(self) -> Tensor:
        """返回 w2c 矩阵（当前约定下）。"""
    
    def inverse(self) -> "CameraPose":
        """高效 SE3 逆（利用 R^T 结构，避免通用 matrix inverse）。
        同时翻转 extrinsic_type 标记。"""

    # ---- 工厂方法：从各种输入格式构造 ----
    @staticmethod
    def from_matrix(mat, convention, extrinsic_type) -> "CameraPose": ...
    
    @staticmethod
    def from_Rt(R, t, convention, extrinsic_type) -> "CameraPose":
        """从 3x3 R + 3 t 构造 4x4。"""
    
    @staticmethod
    def from_qvec_tvec(qvec, tvec, convention=COLMAP, extrinsic_type=W2C) -> "CameraPose":
        """从 COLMAP 风格四元数+平移构造。默认 COLMAP w2c。"""
    
    @staticmethod
    def from_pt3d(R, T) -> "CameraPose":
        """从 PyTorch3D 的 R,T 格式构造（PT3D 的 R 是转置的、轴翻转的）。
        内部处理 PT3D 的特殊矩阵布局，输出标记为 OPENCV c2w。"""

    def __repr__(self):
        return f"CameraPose({self.convention.value}, {self.extrinsic_type.value}, shape={list(self.matrix.shape)})"
```

**转换原理**：大多数约定（OpenCV, COLMAP, OpenGL, Blender, NerfStudio）相机轴只分两组：

- opencv-like（X-right, Y-down, Z-forward）：OpenCV, COLMAP
- opengl-like（X-right, Y-up, Z-backward）：OpenGL, Blender, NerfStudio

两组之间的转换就是右乘（c2w）或左乘（w2c）一个固定的 `diag(1,-1,-1,1)` 矩阵。`CameraPose.to()` 内部查表完成，不需要为每对约定写函数。

### 4. `conversions.py` -- 内部实现（非公共 API）

这个文件是 `CameraPose.to()` 的底层实现，**不对外暴露命名的 pair 函数**。核心内容：

```python
# 查找表：每种约定归到哪一组（相机轴等价类）
_AXIS_GROUP = {
    OPENCV: "opencv",  COLMAP: "opencv",
    OPENGL: "opengl",  BLENDER: "opengl",  NERFSTUDIO: "opengl",
}

# 组间转换矩阵（只有一个，因为只有两组，且转换矩阵是自逆的）
_FLIP = torch.diag(torch.tensor([1., -1., -1., 1.]))

def convert_axes(mat: Tensor, src: CameraConvention, dst: CameraConvention,
                 extrinsic_type: ExtrinsicType) -> Tensor:
    """内部函数：相机轴组间转换。同组 no-op，异组乘 _FLIP。"""
    if _AXIS_GROUP[src] == _AXIS_GROUP[dst]:
        return mat
    if extrinsic_type == ExtrinsicType.C2W:
        return mat @ _FLIP  # c2w: 右乘
    else:
        return _FLIP @ mat  # w2c: 左乘

def se3_inv(mat: Tensor) -> Tensor:
    """高效 SE3 逆。利用 R^T 结构，比 torch.inverse 更快且数值更稳定。
    支持 (*, 4, 4) 和 (*, 3, 4) 输入。"""
    R = mat[..., :3, :3]
    t = mat[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    result = torch.zeros_like(mat[..., :4, :4])
    result[..., :3, :3] = R_inv
    result[..., :3, 3:] = t_inv
    result[..., 3, 3] = 1.0
    return result

def qvec_to_rotmat(qvec) -> Tensor:
    """COLMAP 风格四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    ...
```

**为什么不暴露 `blender_to_opencv()` 等函数**：

- 用户代码应写 `CameraPose(mat, BLENDER, C2W).to(OPENCV).matrix`
- 一个 `.to()` 替代所有 pair 函数，无论今后加多少种约定
- 旧代码中的 `blender2opencv_c2w` 直接替换为这一行

`**se3_inv` 的定位**：

- 短名字，可复用 -- 替代原来的 `closed_form_inverse_se3`
- 既是 `CameraPose.inverse()` 的内部实现，也作为独立工具函数导出（处理裸 Tensor 的场景）
- [vggt/utils/geometry.py](src/model/encoder/vggt/utils/geometry.py) 的 `closed_form_inverse_se3` 改为：`from src.coord import se3_inv as closed_form_inverse_se3`（向后兼容的别名）

### 5. `intrinsics.py` -- 内参转换

统一 `colmap_to_opencv_intrinsics` / `opencv_to_colmap_intrinsics`（当前在 [src/utils/geometry.py](src/utils/geometry.py)、[src/geometry/ptc_geometry.py](src/geometry/ptc_geometry.py)、[src/dataset/shims/geometry_shim.py](src/dataset/shims/geometry_shim.py) 三处重复）。同时支持 torch Tensor 和 numpy array 输入。

### 6. `ply_io.py` -- PLY 读写统一

合并 3 套 PLY 导出，统一约定：

- `export_gaussian_ply(means, scales, rotations, ..., convention=CameraConvention.OPENCV)` -- 明确声明坐标系
- 写入 PLY 的 comment 头中记录坐标系（`comment coordinate_convention=opencv`）
- 读取时解析 comment 头并打印/返回约定信息
- 统一 scale 的 log 处理和 quaternion 的 wxyz 约定
- `load_gaussian_ply(path) -> (GaussianData, CameraConvention)` -- 读取并报告约定

### 7. `_logging.py` -- 运行时可追踪性

```python
import logging
logger = logging.getLogger("coord")

def log_coordinate_op(operation: str, convention: CameraConvention, 
                       extrinsic_type: ExtrinsicType, shape: tuple, context: str = ""):
    logger.info(f"[COORD] {operation} | {convention.value}/{extrinsic_type.value} | shape={shape} | {context}")

def assert_convention(pose: CameraPose, expected_conv: CameraConvention, 
                       expected_type: ExtrinsicType, msg: str = ""):
    assert pose.convention == expected_conv and pose.extrinsic_type == expected_type, \
        f"Expected {expected_conv.value}/{expected_type.value}, got {pose.convention.value}/{pose.extrinsic_type.value}. {msg}"
```

## 集成策略（渐进式）

不一次性重写所有调用方，而是：

1. **Phase 1**：创建 `src/coord/` 模块，含完整 API 和测试
2. **Phase 2**：去重 -- 2 处 `blender2opencv_c2w` 替换为 `CameraPose(...).to(OPENCV)`；`closed_form_inverse_se3` 改为 `se3_inv` 的 re-export 别名；3 处内参转换收敛到 `src/coord/intrinsics`
3. **Phase 3**：数据集边界 -- 在各 dataset loader 的加载链中标注原始约定 -> 输出约定，补充 `dataset_custom.py` 缺失的文档，在 `__getitem__` 添加 debug 级别坐标日志
4. **Phase 4**：PLY / 渲染边界 -- 统一 PLY 读写、在 rendering 入口添加坐标日志
5. **Phase 5**：逐步将 [projection.py](src/geometry/projection.py)、[cam_utils.py](src/misc/cam_utils.py)、[pose.py](src/utils/pose.py) 中的函数代理到 `src/coord/`（保持旧 import 路径兼容）

## 额外建议

- **Cursor Rule 文件**：创建 `.cursor/rules/coordinate-conventions.mdc`，写入本项目的坐标系约定规则 + 各数据集约定速查表，让 AI 在编码时自动遵守
- **单元测试**：为每种约定间的转换写 round-trip 测试（A -> B -> A == identity）
- **ASCII 图注释**：在 `conventions.py` 顶部用 ASCII 图画出各坐标系的轴方向，方便人和 AI 一目了然
- **Custom 数据集文档**：在 `dataset_custom.py` 的 docstring 中明确补充：1) c2w 必须为 OpenCV 相机系；2) K 的像素中心约定（与 meshgrid 整数 u,v 一致）；3) 若来自 COLMAP 需手动减 0.5


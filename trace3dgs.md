# diff-gaussian-rasterization：`trace` 功能说明（3DGS 版）

## 功能是什么？

`trace` 提供一条**与 `diff-surfel-rasterization` 对齐思路**的路径（**不可微**，不参与 autograd）：

- 使用和正常 **3D Gaussian Splatting** 一样的几何：缩放/旋转或预计算 3D 协方差 → **2D conic + opacity** 做 \(\alpha\) 混合。
- 同时正常合成 **RGB 图像**。
- 对图像上的 **`img_sem`（语义特征）**：在 **`img_mask > 0`** 的像素处，按与 RGB 一致的 **`α·T` 权重**，用原子加把语义**累加到每个高斯**，得到 **`gau_sem`**，并维护 **`num_gsem`**（像素级命中计数）。

**不是** surfel / 2DGS 的那种 `transMat` 射线求交几何；只是把「语义回填到高斯」这一层 API 设计成和 surfel 类似，方便你从 anysplat 里迁调用。

---

## 编译与依赖

- 需要 **CUDA** + **已安装带 CUDA 的 PyTorch**。例如：
  ```bash
  cd /path/to/diff-gaussian-rasterization
  pip install --no-build-isolation .
  ```
- **`glm` 头是子模块**，缺了会报 `glm/glm.hpp: No such file or directory`。在仓库根目录：
  ```bash
  git submodule update --init --recursive third_party/glm
  ```
- **`TRACE_CHANNELS` 宏**在 `cuda_rasterizer/config.h`（默认常为 `20`）。改这个数字后必须**重新编译**扩展。

---

## `TRACE_CHANNELS` 与数据形状

Python 里没有单独常量枚举；必须与 **C++ 里编译进去的 `TRACE_CHANNELS`** 一致。

| 符号 | dtype | shape | 说明 |
|------|-------|-------|------|
| `img_sem` | float32 | `(H, W, TRACE_CHANNELS)` | `H/W` 与 `GaussianRasterizationSettings.image_height/width` 一致 |
| `img_mask` | int32 | `(H, W)` | `> 0` 的像素才把该像素的语义写回高斯 |

未传 `img_mask`、但 `img_sem` 非空时，封装里会为 **同一 device** 建全 1 mask。  
若 `img_sem.numel()==0`，语义路径关闭（不向 `img_sem/img_mask` 读数据）。

底层 C++：`rasterize_points.cu` 里对「非空 `img_sem`」会做形状检查。

---

## API

### `GaussianRasterizationSettings`

与同仓库 **`forward`** 相同：`tanfovx / tanfovy`、`bg`、`viewmatrix`、`projmatrix`、`scale_modifier`、`sh_degree`、`campos`、`prefiltered`、`debug`、`image_height`、`image_width` 等。  
你需要用**同一相机、同一分辨率**的一套 setting 来写 `forward`/`trace`。

### `GaussianRasterizer.trace(...)`

语义与原版 `GaussianRasterizer.forward` 的「二选一」规则一致：

- **`shs`** 与 **`colors_precomp`**：必须且只能提供一个。
- **`scales + rotations`** 与 **`cov3D_precomp`**：必须且只能提供一种。

签名（略去默认值）：

```text
trace(self, means3D, means2D, opacities,
      shs=None, colors_precomp=None,
      img_sem=None, img_mask=None,
      scales=None, rotations=None, cov3D_precomp=None)
```

返回值（**5 个**）：

```text
color, gau_depth, gau_sem, num_gsem, radii
```

- `color`: `(NUM_CHANNELS, H, W)`，默认 `NUM_CHANNELS=3`。  
- `gau_depth`: `(P,)`。  
- `gau_sem`: `(P, TRACE_CHANNELS)`。  
- `num_gsem`: `(P,)`，`int32`。  
- `radii`: `(P,)`，`int32`。

底层还会产生 `geomBuffer/binningBuffer/imgBuffer`；当前 Python **不把它们返回给上层**，和 surfel 侧「只拿回主结果」用法一致。

### `trace_gaussians(...)` 顶层函数

适合不想挂 `GaussianRasterizer` 模块时拆开传参；参数表里 `img_semantics`/`img_mask` 与上面的 `img_sem/img_mask` 对应。

---

## `means2D` 说明

沿用原 3DGS 扩展的调用习惯：**仍会传 `means2D`**。CUDA **主路径用 `means3D` + 视图矩阵做投影**，`means2D` 与原版 `forward` 一样保留在接口层；你项目里怎么给 `forward` 的就怎么给 `trace`，避免训练和调试两套不一致。

---

## 梯度 / 用法注意

- `trace`**不能** `.backward()`。应在 `torch.no_grad()` 下调用。
- **`img_sem`、`img_mask`、输出 `gau_*` 等都不建计算图**。  
- 若训练里需要「可微的语义」，要在 PyTorch 里另写 loss，不要指望这个算子反传。

---

## 与 surfel 版对应关系（迁代码时对照）

| 概念 | surfel | 本 3DGS trace |
|------|--------|----------------|
| 几何 | `transMat`、法等 | `scales+rot` 或 `cov3D_precomp` + conic |
| 语义输入 | `(H,W,C)`，`C=TRACE_CHANNELS` | 同款 |
| 输出 | color、gau_depth、gau_sem、num_gsem、radii… | Python 外露这 5 项 |

surfel → 3DGS 迁调用时：**改掉几何相关张量**，`img_sem`/`img_mask` 与「返回的语义/计数」对齐即可。

---

## 完整示例（尽量贴近真实，非两行伪代码）

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

device = torch.device("cuda")

# --------- 下面这些应与你训练 pipeline 一致 ---------
P = 1000
H, W = 480, 640
TRACE_C = 20   # 须与 cuda_rasterizer/config.h 里 TRACE_CHANNELS 一致

means3D = torch.randn(P, 3, device=device) * 0.02
means2D = torch.zeros(P, 3, device=device)   # 与同仓库 forward 相同占位习惯即可
scales = torch.randn(P, 3, device=device).exp() * 0.01
rotations = torch.randn(P, 4, device=device)  # 需归一化四元数以符合规范；此处仅演示调用
rotations = rotations / rotations.norm(dim=-1, keepdim=True)
opacity = torch.sigmoid(torch.randn(P, 1, device=device))

# SH 路径示例：degree=D 时每点系数个数 M = (D+1)^2
sh_degree = 3
M = (sh_degree + 1) ** 2
shs = torch.zeros(P, M, 3, device=device)

viewmatrix = torch.eye(4, device=device)
projmatrix = torch.eye(4, device=device)
campos = viewmatrix.inverse()[3, :3]

raster_settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=0.5,
    tanfovy=0.5,
    bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=sh_degree,
    campos=campos,
    prefiltered=False,
    debug=False,
)

img_sem = torch.randn(H, W, TRACE_C, device=device)
img_mask = torch.ones(H, W, dtype=torch.int32, device=device)

rasterizer = GaussianRasterizer(raster_settings)

with torch.no_grad():
    color, gau_depth, gau_sem, num_gsem, radii = rasterizer.trace(
        means3D,
        means2D,
        opacity,
        shs=shs,
        img_sem=img_sem,
        img_mask=img_mask,
        scales=scales,
        rotations=rotations,
        # cov3D_precomp=... 若用预计算协方差则勿传 scales/rotations
    )

assert color.shape == (3, H, W)
assert gau_depth.shape == (P,)
assert gau_sem.shape == (P, TRACE_C)
assert num_gsem.shape == (P,)
assert radii.shape == (P,)
```

你用 **预计算颜色** 而不是 SH 时，把 `shs` 改成空张量、`colors_precomp` 设为 `(P, 3)`，规则与原版 `GaussianRasterizer.forward` 完全一样。

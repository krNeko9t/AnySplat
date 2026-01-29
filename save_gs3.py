import torch
import numpy as np
from plyfile import PlyData, PlyElement

def save_ply_from_linear(path, means, scales, rotations, opacities, harmonics):
    """
    将线性空间(Linear Space)的 Tensor 保存为标准 3DGS PLY 格式。
    会自动执行 Log 和 Logit 变换。
    """
    
    # 1. 转换为 Numpy
    def to_numpy(x):
        return x.detach().cpu().numpy().astype(np.float32)

    xyz = to_numpy(means)
    s_linear = to_numpy(scales)      # 假设是真实的物理尺寸 ( > 0 )
    r = to_numpy(rotations)          # (w, x, y, z)
    o_linear = to_numpy(opacities)   # 假设是 0~1 之间的透明度
    sh = to_numpy(harmonics)

    N = xyz.shape[0]
    print(f"转换并保存 {N} 个点...")

    # --- 关键修复：空间转换 ---
    
    # 1. 处理 Scale: Linear -> Log Space
    # 必须防止 log(0) 或负数
    print("Converting Scale to Log Space...")
    s_linear = np.clip(s_linear, 1e-6, 1e8) # 甚至可以 clip 到更小的范围
    s_log = np.log(s_linear) 

    # 2. 处理 Opacity: Linear -> Logit Space (Inverse Sigmoid)
    # logit(p) = log(p / (1 - p))
    # 必须防止 0 和 1，否则会出现 -inf / +inf
    print("Converting Opacity to Logit Space...")
    o_linear = np.clip(o_linear, 1e-4, 1.0 - 1e-4) # 限制在 0.0001 ~ 0.9999
    o_logit = np.log(o_linear / (1.0 - o_linear))

    # 3. 处理 Rotation: 归一化
    # 即使模型输出看起来是对的，存之前最好 normalize 一下
    print("Normalizing Rotations...")
    r_norm = np.linalg.norm(r, axis=1, keepdims=True) + 1e-8
    r = r / r_norm

    # --- 下面是标准的构建过程 ---
    
    # 维度处理
    if o_logit.ndim == 1: o_logit = o_logit[:, None]
    
    # SH 处理 (transpose if needed)
    if sh.shape[1] == 3 and sh.shape[2] > 3:
        sh = sh.transpose(0, 2, 1)
    
    f_dc = sh[:, 0, :].reshape(N, 3)
    # 处理高阶 SH
    if sh.shape[1] > 1:
        f_rest = sh[:, 1:, :].reshape(N, -1)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    # 构建属性字典
    attrs = {}
    attrs['x'], attrs['y'], attrs['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    attrs['nx'], attrs['ny'], attrs['nz'] = 0, 0, 0
    
    for i in range(3): attrs[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(f_rest.shape[1]): attrs[f'f_rest_{i}'] = f_rest[:, i]
    
    # 注意：这里存的是转换后的 o_logit 和 s_log
    attrs['opacity'] = o_logit[:, 0]
    for i in range(3): attrs[f'scale_{i}'] = s_log[:, i]
    for i in range(4): attrs[f'rot_{i}'] = r[:, i]

    # 按官方顺序排序
    sorted_keys = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] + 
        [f'f_dc_{i}' for i in range(3)] + 
        [f'f_rest_{i}' for i in range(f_rest.shape[1])] + 
        ['opacity'] + 
        [f'scale_{i}' for i in range(3)] + 
        [f'rot_{i}' for i in range(4)]
    )

    dtype_list = [(k, 'f4') for k in sorted_keys]
    elements = np.empty(N, dtype=dtype_list)
    for k in sorted_keys:
        elements[k] = attrs[k]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"✅ 保存成功 (已应用 Log/Logit 变换): {path}")

# 使用方法：
# save_ply_from_linear("fixed_output.ply", means, scales, rots, opacities, shs)
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def save_tensors_to_ply(path, means, scales, rotations, opacities, harmonics):
    """
    将 Gaussian Tensors 保存为 .ply 文件。
    
    参数 (均为 Tensor):
    path (str): 保存路径
    means: [N, 3]
    scales: [N, 3] (建议是 Log Space)
    rotations: [N, 4] (w, x, y, z)
    opacities: [N, 1] (建议是 Logit Space)
    harmonics: [N, K, 3] 或 [N, 3, K] 球谐系数
    """
    
    # --- 1. 辅助函数：转 Numpy ---
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # 转换所有数据
    xyz = to_numpy(means)
    s = to_numpy(scales)
    r = to_numpy(rotations)
    o = to_numpy(opacities)
    sh = to_numpy(harmonics)
    
    N = xyz.shape[0]
    print(f"Processing {N} points...")

    # --- 2. 数据清洗与形状检查 ---
    # 确保 Opacity 是 [N, 1] 而不是 [N]
    if o.ndim == 1: o = o[:, None]
    
    # 处理 SH (球谐): 
    # 假设输入形状是 [N, K, 3] (N, 系数数量, RGB通道)
    # 官方代码中有时是 features_dc [N, 1, 3] 和 features_rest [N, 15, 3]
    # 这里假设 harmonics 已经拼接好了。
    
    # 如果你的 SH 维度是 [N, 3, K] (通道在前)，需要 transpose 过来
    if sh.shape[1] == 3 and sh.shape[2] > 3: 
        sh = sh.transpose(0, 2, 1)
        
    # 分离 DC (f_dc) 和 Rest (f_rest)
    # f_dc: [N, 3] -> 对应 ply 属性 f_dc_0, f_dc_1, f_dc_2
    f_dc = sh[:, 0, :].reshape(N, 3)
    
    # f_rest: [N, (K-1)*3] -> 对应 ply 属性 f_rest_0 ... f_rest_44
    if sh.shape[1] > 1:
        f_rest = sh[:, 1:, :].reshape(N, -1)
    else:
        f_rest = np.zeros((N, 0)) # 只有 0 阶的情况

    # --- 3. 构建 PLY 数据结构 ---
    # 定义属性头
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ]
    
    # 添加 f_dc
    for i in range(3): dtype_list.append((f'f_dc_{i}', 'f4'))
    # 添加 f_rest
    for i in range(f_rest.shape[1]): dtype_list.append((f'f_rest_{i}', 'f4'))
    # 添加 opacity
    dtype_list.append(('opacity', 'f4'))
    # 添加 scale
    for i in range(3): dtype_list.append((f'scale_{i}', 'f4'))
    # 添加 rot
    for i in range(4): dtype_list.append((f'rot_{i}', 'f4'))

    # 创建 buffer
    elements = np.empty(N, dtype=dtype_list)
    
    # 填充基础几何
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = 0, 0, 0 # 法线通常填0
    
    # 填充颜色 (SH)
    for i in range(3):
        elements[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(f_rest.shape[1]):
        elements[f'f_rest_{i}'] = f_rest[:, i]
        
    # 填充物理属性
    elements['opacity'] = o[:, 0]
    for i in range(3): elements[f'scale_{i}'] = s[:, i]
    for i in range(4): elements[f'rot_{i}'] = r[:, i]

    # --- 4. 写入 ---
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved to {path}")

# 使用示例（假设你的变量名如下）
# save_tensors_to_ply("result.ply", means, scales, rotations, opacities, harmonics)
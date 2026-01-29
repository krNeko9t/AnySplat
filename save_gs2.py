import torch
import numpy as np
from plyfile import PlyData, PlyElement

def save_gaussian_to_ply_robust(path, means, scales, rotations, opacities, harmonics):
    """
    鲁棒的 PLY 保存函数，完全对齐官方格式，并包含 NaN/Inf 检查。
    """
    
    # 1. 转换为 Numpy & 强制 float32 (PLY 标准格式)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return x.astype(np.float32)

    xyz = to_numpy(means)      # [N, 3]
    s = to_numpy(scales)       # [N, 3]
    r = to_numpy(rotations)    # [N, 4]
    o = to_numpy(opacities)    # [N, 1]
    sh = to_numpy(harmonics)   # [N, K, 3] 

    N = xyz.shape[0]
    print(f"正在处理 {N} 个 Gaussian 点...")

    # --- [关键步骤]：数据健康检查 ---
    # 很多查看器遇到 NaN/Inf 会直接崩溃
    print("正在检查数据有效性 (NaN/Inf)...")
    if not np.isfinite(xyz).all(): raise ValueError("❌ Means (XYZ) 包含 NaN 或 Inf")
    if not np.isfinite(s).all():   raise ValueError("❌ Scales 包含 NaN 或 Inf")
    if not np.isfinite(r).all():   raise ValueError("❌ Rotations 包含 NaN 或 Inf")
    if not np.isfinite(o).all():   raise ValueError("❌ Opacities 包含 NaN 或 Inf")
    if not np.isfinite(sh).all():  raise ValueError("❌ SHs 包含 NaN 或 Inf")
    print("✅ 数据数值检查通过")

    # --- 2. 维度标准化 ---
    if o.ndim == 1: o = o[:, None]
    
    # 处理 SH: 确保是 [N, (degrees+1)^2, 3]
    # 如果你的 SH 是 [N, 3, K] (Channel first)，需要转置
    if sh.shape[1] == 3 and sh.shape[2] > 3:
        sh = sh.transpose(0, 2, 1)
        
    # 分离 DC 和 Rest
    # f_dc: [N, 1, 3] -> flatten -> [N, 3]
    f_dc = sh[:, 0, :].reshape(N, 3)
    
    # f_rest: [N, 15, 3] -> flatten -> [N, 45] (如果是3阶)
    if sh.shape[1] > 1:
        f_rest = sh[:, 1:, :].reshape(N, -1)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    # --- 3. 构建结构化数据 (完全对齐官方属性名) ---
    # 使用字典构建，避免手动对齐索引的错误
    attrs = {}
    
    # 几何
    attrs['x'] = xyz[:, 0]
    attrs['y'] = xyz[:, 1]
    attrs['z'] = xyz[:, 2]
    attrs['nx'] = np.zeros(N, dtype=np.float32)
    attrs['ny'] = np.zeros(N, dtype=np.float32)
    attrs['nz'] = np.zeros(N, dtype=np.float32)
    
    # 颜色 (f_dc_0, f_dc_1, f_dc_2)
    for i in range(3):
        attrs[f'f_dc_{i}'] = f_dc[:, i]
        
    # 颜色 (f_rest_0 ... f_rest_N)
    for i in range(f_rest.shape[1]):
        attrs[f'f_rest_{i}'] = f_rest[:, i]
        
    # 物理属性
    attrs['opacity'] = o[:, 0]
    
    for i in range(3):
        attrs[f'scale_{i}'] = s[:, i]
        
    for i in range(4):
        attrs[f'rot_{i}'] = r[:, i]

    # --- 4. 转换为 plyfile 需要的结构化数组 ---
    # 按官方顺序定义属性名 (顺序很重要！)
    # 官方顺序: xyz -> normals -> f_dc -> f_rest -> opacity -> scale -> rot
    sorted_keys = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] + 
        [f'f_dc_{i}' for i in range(3)] + 
        [f'f_rest_{i}' for i in range(f_rest.shape[1])] + 
        ['opacity'] + 
        [f'scale_{i}' for i in range(3)] + 
        [f'rot_{i}' for i in range(4)]
    )
    
    # 创建 dtype
    dtype_list = [(k, 'f4') for k in sorted_keys]
    elements = np.empty(N, dtype=dtype_list)
    
    # 填充数据
    for k in sorted_keys:
        elements[k] = attrs[k]

    # --- 5. 写入 ---
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"✅ 成功保存: {path}")

# --- 测试调用 ---
# try:
#     save_gaussian_to_ply_robust("fixed.ply", means, scales, rots, opacities, shs)
# except ValueError as e:
#     print(e)
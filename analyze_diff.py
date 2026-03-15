import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
volume_dir = os.path.join(base_dir, "volume")
output_dir = os.path.join(base_dir, "converted")

# 对比函数 - 尝试不同的字节序
def compare_with_endian(vol_file, conv_file, name):
    """尝试不同的字节序对比"""
    print(f"\n{'='*60}")
    print(f"尝试不同字节序对比: {name}")
    print(f"{'='*60}")
    
    # 读取原始dat文件
    original = np.fromfile(vol_file, dtype=np.float32)
    converted = np.fromfile(conv_file, dtype=np.float32)
    
    print(f"原始数据 (float32): shape={original.shape}, min={original.min():.4f}, max={original.max():.4f}")
    print(f"转换数据 (float32): shape={converted.shape}, min={converted.min():.4f}, max={converted.max():.4f}")
    
    # 尝试字节交换
    swapped = converted.byteswap()
    print(f"\n字节交换后: min={swapped.min():.4f}, max={swapped.max():.4f}")
    
    if np.array_equal(original, swapped):
        print("✓ 字节交换后完全一致！")
        return True
    
    # 尝试不同的dtype
    dtypes_to_try = [
        ('float64', np.float64),
        ('int32', np.int32),
        ('uint32', np.uint32),
        ('int16', np.int16),
        ('uint16', np.uint16),
    ]
    
    for dtype_name, dtype in dtypes_to_try:
        # 读取为不同类型
        orig_data = np.fromfile(vol_file, dtype=dtype)
        conv_data = np.fromfile(conv_file, dtype=dtype)
        
        if np.array_equal(orig_data, conv_data):
            print(f"✓ 使用 {dtype_name} 完全一致！")
            return True
        
        # 尝试字节交换
        conv_swapped = conv_data.byteswap()
        if np.array_equal(orig_data, conv_swapped):
            print(f"✓ 使用 {dtype_name} + 字节交换 完全一致！")
            return True
    
    # 分析差异模式
    diff = np.abs(original - converted)
    print(f"\n差异分析 (float32):")
    print(f"  最大差异: {diff.max()}")
    print(f"  平均差异: {diff.mean()}")
    print(f"  差异标准差: {diff.std()}")
    
    # 检查是否是有规律的差异
    unique_diffs = np.unique(diff)
    print(f"  独特差异值数量: {len(unique_diffs)}")
    if len(unique_diffs) < 20:
        print(f"  独特差异值: {unique_diffs[:20]}")
    
    return False

# 对比两个文件
print("分析数据差异...")

# H2 对比
result_h2 = compare_with_endian(
    os.path.join(volume_dir, "H2.dat"),
    os.path.join(output_dir, "H2_converted.dat"),
    "H2"
)

# PD 对比
result_pd = compare_with_endian(
    os.path.join(volume_dir, "PD.dat"),
    os.path.join(output_dir, "PD_converted.dat"),
    "PD"
)

# 额外分析：检查原始volume文件的数据范围
print(f"\n{'='*60}")
print("volume文件夹中原始文件分析")
print(f"{'='*60}")

for fname in ["H2.dat", "PD.dat"]:
    fpath = os.path.join(volume_dir, fname)
    data = np.fromfile(fpath, dtype=np.float32)
    print(f"{fname}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

# 检查npy文件的数据范围
print(f"\n{'='*60}")
print("npy源文件数据范围")
print(f"{'='*60}")

for fname in ["volRendering_H2.npy", "volRendering_PD.npy"]:
    fpath = os.path.join(base_dir, fname)
    data = np.load(fpath)
    print(f"{fname}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

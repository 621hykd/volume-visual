import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
npy_dir = base_dir
volume_dir = os.path.join(base_dir, "volume")
output_dir = os.path.join(base_dir, "converted")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 转换函数：将npy转换为dat（反向归一化到0-255范围）
def convert_npy_to_dat(npy_path, dat_path, target_min=0, target_max=255):
    """
    将npy文件转换为dat文件
    参数:
        npy_path: 输入的npy文件路径
        dat_path: 输出的dat文件路径
        target_min: 目标数据的最小值（默认0）
        target_max: 目标数据的最大值（默认255）
    """
    # 读取npy文件
    data = np.load(npy_path)
    
    # 获取源数据的范围
    src_min = data.min()
    src_max = data.max()
    
    print(f"源数据范围: [{src_min}, {src_max}]")
    print(f"目标数据范围: [{target_min}, {target_max}]")
    
    # 线性归一化到目标范围
    # 从 [src_min, src_max] 映射到 [target_min, target_max]
    normalized_data = (data - src_min) / (src_max - src_min) * (target_max - target_min) + target_min
    
    # 转换为float32并保存为二进制
    normalized_data = normalized_data.astype(np.float32)
    normalized_data.tofile(dat_path)
    
    print(f"转换完成: {os.path.basename(npy_path)} -> {os.path.basename(dat_path)}")
    print(f"  数据形状: {normalized_data.shape}")
    print(f"  转换后范围: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    print(f"  文件大小: {os.path.getsize(dat_path)} bytes")

# 对比函数
def compare_dat_files(file1, file2, name1="file1", name2="file2"):
    """对比两个dat文件"""
    # 读取二进制数据
    data1 = np.fromfile(file1, dtype=np.float32)
    data2 = np.fromfile(file2, dtype=np.float32)
    
    print(f"\n{'='*60}")
    print(f"对比结果: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"  {name1} 大小: {len(data1)} 元素, 范围: [{data1.min():.4f}, {data1.max():.4f}]")
    print(f"  {name2} 大小: {len(data2)} 元素, 范围: [{data2.min():.4f}, {data2.max():.4f}]")
    
    if len(data1) != len(data2):
        print(f"  ❌ 大小不一致！")
        return False
    
    # 检查是否完全一致
    if np.array_equal(data1, data2):
        print(f"  ✓✓✓ 完全一致！✓✓✓")
        return True
    
    # 计算差异
    diff = np.abs(data1 - data2)
    print(f"  最大差异: {diff.max()}")
    print(f"  平均差异: {diff.mean()}")
    print(f"  差异标准差: {diff.std()}")
    
    # 检查差异占比
    matching = np.sum(diff == 0)
    print(f"  匹配元素: {matching}/{len(data1)} ({matching/len(data1)*100:.2f}%)")
    
    return False

# 执行转换
print("=" * 60)
print("开始转换 npy -> dat（包含反向归一化）")
print("=" * 60)

# 转换 H2（使用volume中原始数据的范围作为参考）
print("\n--- 转换 H2 ---")
convert_npy_to_dat(
    os.path.join(npy_dir, "volRendering_H2.npy"),
    os.path.join(output_dir, "H2_converted.dat"),
    target_min=0,
    target_max=255
)

# 转换 PD
print("\n--- 转换 PD ---")
convert_npy_to_dat(
    os.path.join(npy_dir, "volRendering_PD.npy"),
    os.path.join(output_dir, "PD_converted.dat"),
    target_min=0,
    target_max=255
)

# 对比转换后的文件和volume中的原始文件
print("\n" + "=" * 60)
print("对比转换结果")
print("=" * 60)

# 对比 H2
result_h2 = compare_dat_files(
    os.path.join(volume_dir, "H2.dat"),
    os.path.join(output_dir, "H2_converted.dat"),
    "volume/H2.dat",
    "converted/H2_converted.dat"
)

# 对比 PD
result_pd = compare_dat_files(
    os.path.join(volume_dir, "PD.dat"),
    os.path.join(output_dir, "PD_converted.dat"),
    "volume/PD.dat",
    "converted/PD_converted.dat"
)

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"H2 转换结果: {'✓ 一致' if result_h2 else '❌ 不一致'}")
print(f"PD 转换结果: {'✓ 一致' if result_pd else '❌ 不一致'}")

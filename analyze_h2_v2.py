import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
volume_dir = os.path.join(base_dir, "volume")

# 读取数据
npy_h2 = np.load(os.path.join(base_dir, "volRendering_H2.npy"))
dat_h2 = np.fromfile(os.path.join(volume_dir, "H2.dat"), dtype=np.float32)

print("="*60)
print("深入分析 H2 数据结构")
print("="*60)

# 分析npy数据分布
print("\n--- npy数据分布 ---")
print(f"总元素数: {len(npy_h2)}")
print(f"唯一值数量: {len(np.unique(npy_h2))}")
print(f"最小值: {npy_h2.min()}")
print(f"最大值: {npy_h2.max()}")

# 统计各值数量
unique, counts = np.unique(npy_h2, return_counts=True)
print(f"\n前10个最常见的值:")
sorted_idx = np.argsort(-counts)[:10]
for i in sorted_idx:
    print(f"  值={unique[i]:.6f}, 数量={counts[i]} ({counts[i]/len(npy_h2)*100:.2f}%)")

# 分析dat数据分布
print("\n--- dat数据分布 ---")
print(f"总元素数: {len(dat_h2)}")
print(f"唯一值数量: {len(np.unique(dat_h2))}")
print(f"最小值: {dat_h2.min()}")
print(f"最大值: {dat_h2.max()}")

# 统计各值数量
unique_dat, counts_dat = np.unique(dat_h2, return_counts=True)
print(f"\n前10个最常见的值:")
sorted_idx_dat = np.argsort(-counts_dat)[:10]
for i in sorted_idx_dat:
    print(f"  值={unique_dat[i]:.6f}, 数量={counts_dat[i]} ({counts_dat[i]/len(dat_h2)*100:.2f}%)")

# 关键发现：检查是否数据有特定模式
print("\n--- 检查数据模式 ---")

# 检查npy中接近-1的值占比
near_neg1 = np.sum(np.isclose(npy_h2, -1.0))
print(f"npy中接近-1.0的值: {near_neg1} ({near_neg1/len(npy_h2)*100:.2f}%)")

# 检查dat中为0的值
dat_zero = np.sum(dat_h2 == 0)
print(f"dat中为0的值: {dat_zero} ({dat_zero/len(dat_h2)*100:.2f}%)")

# 尝试另一种思路：可能npy数据和dat数据不是一一对应关系
# 可能npy数据本身就是一种不同的表示

print("\n--- 尝试直接保存为uint8 ---")
# 尝试将npy数据直接保存为uint8（不归一化）
data_uint8 = (npy_h2 * 255).astype(np.uint8)
data_uint8.tofile(os.path.join(base_dir, "converted", "H2_uint8.dat"))

# 读取并对比
converted_uint8 = np.fromfile(os.path.join(base_dir, "converted", "H2_uint8.dat"), dtype=np.uint8).astype(np.float32)
diff_uint8 = np.abs(converted_uint8 - dat_h2)
print(f"uint8转换后最大差异: {diff_uint8.max()}")
print(f"uint8转换后平均差异: {diff_uint8.mean():.4f}")
print(f"完全匹配: {np.sum(diff_uint8 == 0)}/{len(diff_uint8)}")

# 四舍五入后匹配
rounded = np.round(converted_uint8)
match_rounded = np.sum(rounded == dat_h2)
print(f"四舍五入后匹配: {match_rounded}/{len(rounded)}")

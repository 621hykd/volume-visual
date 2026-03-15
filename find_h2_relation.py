import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
volume_dir = os.path.join(base_dir, "volume")

# 读取数据
npy_h2 = np.load(os.path.join(base_dir, "volRendering_H2.npy"))
dat_h2 = np.fromfile(os.path.join(volume_dir, "H2.dat"), dtype=np.float32)

print("="*60)
print("寻找 H2 的正确转换关系")
print("="*60)

# 关键发现：npy中-1对应dat中0
# 尝试：dat = npy + 1，然后缩放到255

# 方法1：dat = (npy + 1) * 127.5
method1 = (npy_h2 + 1) * 127.5
print(f"\n方法1: (npy + 1) * 127.5")
print(f"  范围: [{method1.min():.4f}, {method1.max():.4f}]")
diff1 = np.abs(method1 - dat_h2)
print(f"  最大差异: {diff1.max():.4f}")
print(f"  平均差异: {diff1.mean():.4f}")
print(f"  完全匹配: {np.sum(diff1 == 0)}/{len(diff1)}")

# 方法2：考虑到dat中非零值有微小偏移
# 假设：dat = (npy + 1) * 255 / 2
method2 = (npy_h2 + 1) * 127.5
# 找到最佳的偏移量
offset = 0
best_match = 0
for offset_test in np.linspace(0, 2, 100):
    test_data = method2 + offset_test
    match = np.sum(np.abs(test_data - dat_h2) < 0.001)
    if match > best_match:
        best_match = match
        offset = offset_test

print(f"\n方法2: (npy + 1) * 127.5 + {offset:.4f}")
method2_adj = method2 + offset
print(f"  范围: [{method2_adj.min():.4f}, {method2_adj.max():.4f}]")
diff2 = np.abs(method2_adj - dat_h2)
print(f"  最大差异: {diff2.max():.4f}")
print(f"  平均差异: {diff2.mean():.4f}")
print(f"  完全匹配: {np.sum(diff2 < 0.001)}/{len(diff2)}")

# 方法3：直接分析对应关系
# npy=-1 → dat=0
# npy=-0.99... → dat=1

# 找出npy中非-1的值并对应到dat
npy_nonzero = npy_h2[npy_h2 != -1.0]
dat_nonzero = dat_h2[dat_h2 != 0.0]

# 需要确保对应关系正确
# 用排序来匹配
npy_sorted = np.sort(npy_nonzero)
dat_sorted = np.sort(dat_nonzero)

print(f"\n非零值分析:")
print(f"  npy非零值: {len(npy_sorted)}")
print(f"  dat非零值: {len(dat_sorted)}")

# 尝试直接用排序后的数据进行映射
# 创建一个映射
npy_vals = np.unique(npy_h2)
dat_vals = np.unique(dat_h2)

print(f"\n唯一值对比:")
print(f"  npy唯一值范围: [{npy_vals.min():.6f}, {npy_vals.max():.6f}]")
print(f"  dat唯一值范围: [{dat_vals.min():.6f}, {dat_vals.max():.6f}]")

# 关键：检查是否只是简单的平移
# npy值 + 1 ≈ dat值/255 * 2 - 1 ?
# 或者 dat = (npy + 1) / 2 * 255 + 某个偏移

# 让我尝试找出精确的线性关系
# 假设：dat = a * (npy + 1) + b
# 从已知的对应关系：npy=-1 → dat=0
# 所以：0 = a * 0 + b → b = 0

# 尝试：dat = a * (npy + 1)
# 对于npy接近1的值，应该对应dat接近255
# 1 = a * 2 → a = 0.5
# 所以：dat = 0.5 * (npy + 1) = npy/2 + 0.5

method3 = npy_h2 / 2 + 0.5
method3 = method3 * 255

print(f"\n方法3: (npy/2 + 0.5) * 255")
print(f"  范围: [{method3.min():.4f}, {method3.max():.4f}]")
diff3 = np.abs(method3 - dat_h2)
print(f"  最大差异: {diff3.max():.4f}")
print(f"  平均差异: {diff3.mean():.4f}")
print(f"  完全匹配: {np.sum(diff3 == 0)}/{len(diff3)}")

# 既然PD一致了，让我看看PD的转换关系是怎么成功的
print("\n" + "="*60)
print("检查 PD 为什么一致")
print("="*60)

npy_pd = np.load(os.path.join(base_dir, "volRendering_PD.npy"))
dat_pd = np.fromfile(os.path.join(volume_dir, "PD.dat"), dtype=np.float32)

print(f"\nnpy_pd范围: [{npy_pd.min():.4f}, {npy_pd.max():.4f}]")
print(f"dat_pd范围: [{dat_pd.min():.4f}, {dat_pd.max():.4f}]")

# PD的转换是直接线性映射
pd_converted = (npy_pd - npy_pd.min()) / (npy_pd.max() - npy_pd.min()) * 255
print(f"\n直接归一化后范围: [{pd_converted.min():.4f}, {pd_converted.max():.4f}]")
pd_match = np.sum(pd_converted == dat_pd)
print(f"与dat_pd完全匹配: {pd_match}/{len(pd_converted)}")

# H2的关键是大部分值是-1，需要特殊处理
print("\n" + "="*60)
print("H2 特殊处理")
print("="*60)

# H2中-1的值很多（40%+），这些对应dat中的0
# 对于非-1的值，需要找到正确的映射

# 先把-1转为0
h2_temp = npy_h2.copy()
h2_temp[h2_temp == -1.0] = 0

# 对非零值进行归一化
nonzero_mask = h2_temp != 0
if np.sum(nonzero_mask) > 0:
    nonzero_vals = h2_temp[nonzero_mask]
    # 归一化到0-255
    normalized = (nonzero_vals - nonzero_vals.min()) / (nonzero_vals.max() - nonzero_vals.min()) * 255
    
    # 创建完整数组
    result = np.zeros_like(h2_temp)
    result[nonzero_mask] = normalized
    
    print(f"特殊处理后范围: [{result.min():.4f}, {result.max():.4f}]")
    diff_special = np.abs(result - dat_h2)
    print(f"  最大差异: {diff_special.max():.4f}")
    print(f"  平均差异: {diff_special.mean():.4f}")
    print(f"  完全匹配: {np.sum(diff_special == 0)}/{len(diff_special)}")

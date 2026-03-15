import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
volume_dir = os.path.join(base_dir, "volume")
output_dir = os.path.join(base_dir, "converted")

# 读取数据
npy_h2 = np.load(os.path.join(base_dir, "volRendering_H2.npy"))
dat_h2 = np.fromfile(os.path.join(volume_dir, "H2.dat"), dtype=np.float32)

print("="*60)
print("H2 转换 - 四舍五入测试")
print("="*60)

# 使用 (npy + 1) * 127.5 转换
converted = (npy_h2 + 1) * 127.5

print(f"转换后范围: [{converted.min():.4f}, {converted.max():.4f}]")

# 尝试不同的取整方式
print("\n--- 测试不同的取整方式 ---")

# 1. 直接保存
diff_raw = np.abs(converted - dat_h2)
print(f"1. 直接保存: 完全匹配={np.sum(diff_raw == 0)}/{len(diff_raw)}, 最大差异={diff_raw.max():.4f}")

# 2. 四舍五入到整数
rounded = np.round(converted).astype(np.float32)
diff_round = np.abs(rounded - dat_h2)
print(f"2. 四舍五入: 完全匹配={np.sum(diff_round == 0)}/{len(diff_round)}, 最大差异={diff_round.max():.4f}")

# 3. floor后+1（处理-1的情况）
floored = np.floor(converted)
floored[converted > floored] += 1  # 对非整数部分+1
floored = floored.astype(np.float32)
diff_floor = np.abs(floored - dat_h2)
print(f"3. floor+1: 完全匹配={np.sum(diff_floor == 0)}/{len(diff_floor)}, 最大差异={diff_floor.max():.4f}")

# 4. 仔细检查dat中的非零值到底是什么
print("\n--- 详细检查dat中的非零值 ---")
nonzero_dat = dat_h2[dat_h2 != 0]
print(f"非零值数量: {len(nonzero_dat)}")
print(f"非零值范围: [{nonzero_dat.min():.6f}, {nonzero_dat.max():.6f}]")
print(f"非零值唯一数: {len(np.unique(nonzero_dat))}")

# 看看这些非零值减去1后是什么
nonzero_minus1 = nonzero_dat - 1
print(f"减1后范围: [{nonzero_minus1.min():.6f}, {nonzero_minus1.max():.6f}]")

# 对应的npy值
nonzero_npy = npy_h2[npy_h2 != -1.0]
print(f"npy非零值范围: [{nonzero_npy.min():.6f}, {nonzero_npy.max():.6f}]")

# 5. 尝试直接匹配位置
print("\n--- 位置匹配分析 ---")
# 找出npy中-1的位置
npy_neg1_pos = np.where(npy_h2 == -1.0)[0]
dat_zero_pos = np.where(dat_h2 == 0)[0]
print(f"npy中-1的位置数: {len(npy_neg1_pos)}")
print(f"dat中0的位置数: {len(dat_zero_pos)}")
print(f"位置是否一致: {np.array_equal(npy_neg1_pos, dat_zero_pos)}")

# 6. 也许转换时需要用到不同的缩放
# 让我看看npy中的非-1值与dat中的非0值的精确对应
print("\n--- 精确对应关系分析 ---")

# 排序后比较
npy_sorted = np.sort(npy_h2[npy_h2 != -1])
dat_sorted = np.sort(dat_h2[dat_h2 != 0])

print(f"npy排序后前5个: {npy_sorted[:5]}")
print(f"dat排序后前5个: {dat_sorted[:5]}")

# 检查dat值是否 = npy值*k + offset
# 使用线性回归
from numpy.polynomial import polynomial as P
# dat ≈ a * npy + b (对于非特殊值)

mask_npy = npy_h2 != -1
mask_dat = dat_h2 != 0

# 只有两边都满足条件的位置才能对应
# 假设位置一一对应
a_fit = np.cov(npy_h2[mask_npy], dat_h2[mask_dat])[0,1] / np.var(npy_h2[mask_npy])
b_fit = dat_h2[mask_dat].mean() - a_fit * npy_h2[mask_npy].mean()

print(f"\n线性拟合: dat = {a_fit:.6f} * npy + {b_fit:.6f}")

converted_fit = a_fit * npy_h2 + b_fit
diff_fit = np.abs(converted_fit - dat_h2)
print(f"拟合后完全匹配: {np.sum(diff_fit == 0)}/{len(diff_fit)}")
print(f"拟合后最大差异: {diff_fit.max():.6f}")

# 7. 检查是否dat中的值是转换时的四舍五入结果
# 尝试：dat = round((npy + 1) * 127.5)
rounded_method = np.round((npy_h2 + 1) * 127.5)
diff_rounded_method = np.abs(rounded_method - dat_h2)
print(f"\nround((npy+1)*127.5)完全匹配: {np.sum(diff_rounded_method == 0)}/{len(diff_rounded_method)}")

# 关键发现！
# 让我检查dat中值=1的情况对应npy中什么值
print("\n--- 关键发现 ---")
dat_one_pos = np.where(np.isclose(dat_h2, 1.0, atol=0.5))[0]
print(f"dat中值≈1的位置数: {len(dat_one_pos)}")
if len(dat_one_pos) > 0:
    npy_at_dat1 = npy_h2[dat_one_pos[:100]]  # 取前100个
    print(f"对应npy值: {npy_at_dat1[:10]}")
    print(f"对应npy值范围: [{npy_at_dat1.min():.6f}, {npy_at_dat1.max():.6f}]")
    print(f"期望值 (x+1)*127.5=1: x = -0.99215686")

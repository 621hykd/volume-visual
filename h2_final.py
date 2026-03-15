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
print("H2 最终解决方案")
print("="*60)

# 基础转换
converted = (npy_h2 + 1) / 2 * 255

# 检查差异模式
diff = converted - dat_h2

print("差异分析:")
print(f"  diff=0: {np.sum(diff == 0)}")
print(f"  diff=1: {np.sum(diff == 1)}")
print(f"  diff=-1: {np.sum(diff == -1)}")
print(f"  diff接近1: {np.sum(np.isclose(diff, 1, atol=0.1))}")

# 思路1：转换后的值+1后是否等于dat？
# 即：dat = round(converted) + 1 对某些值成立？
plus1_check = np.round(converted) + 1 - dat_h2
print(f"\nround(converted)+1 - dat:")
print(f"  =0: {np.sum(plus1_check == 0)}")

# 思路2：也许问题是dat中的值是经过某种特殊处理
# 检查dat中每个值对应的npy值

# 方法：对每个唯一的dat值，找出对应的npy值范围
unique_dat = np.unique(dat_h2)
print(f"\ndat唯一值数量: {len(unique_dat)}")

# 让我检查是否存在一个精确的一一映射
# 对于每个位置(i)，找npy[i]到dat[i]的映射

# 关键：dat[i] ≈ round((npy[i] + 1) * 127.5)
# 但有些位置会有1的偏差

# 解决方案：
# 如果 round((npy+1)*127.5) + 1 == dat，则保留
# 否则用 round((npy+1)*127.5)

converted_final = np.round(converted).astype(np.float32)

# 检查修正
# 对于转换后值 < dat的情况（diff = -1）
mask_lower = converted_final < dat_h2
mask_higher = converted_final > dat_h2

print(f"\n修正分析:")
print(f"  转换值 < dat: {np.sum(mask_lower)}")
print(f"  转换值 > dat: {np.sum(mask_higher)}")

# 尝试：+1 修正
corrected = converted_final.copy()
corrected[mask_lower] += 1  # 对偏低的加1

diff_corrected = np.abs(corrected - dat_h2)
print(f"\n修正后完全匹配: {np.sum(diff_corrected == 0)}/{len(diff_corrected)}")

# 让我直接看另一种可能：dat原始数据是怎么生成的
# 也许不是从npy转换来的？

# 检查dat值与npy值的关系
print("\n" + "="*60)
print("反向验证：dat -> npy")
print("="*60)

# 假设 dat = (npy + 1) * 127.5
# 则 npy = dat / 127.5 - 1
recovered_npy = dat_h2 / 127.5 - 1

print(f"恢复的npy范围: [{recovered_npy.min():.6f}, {recovered_npy.max():.6f}]")

# 检查恢复后的npy与原始npy的差异
diff_recovered = np.abs(recovered_npy - npy_h2)
print(f"恢复后与原npy差异: max={diff_recovered.max():.6f}, mean={diff_recovered.mean():.6f}")

# 精确匹配的位置
exact_match = np.sum(diff_recovered < 0.001)
print(f"差异<0.001的位置: {exact_match}/{len(diff_recovered)}")

# 最终方案：既然无法完全匹配，检查是否能通过位置匹配
print("\n" + "="*60)
print("最终结论")
print("="*60)

print("""
分析结果：
- npy源数据范围: [-1.0, 1.0]（归一化数据）
- dat目标数据范围: [0, 255]（原始灰度值）

转换公式测试：
1. (npy + 1) / 2 * 255: 14928047/36902400 匹配 (40.45%)
2. 线性归一化到[0,255]: 14928047/36902400 匹配

问题：无法100%匹配，可能原因：
1. npy数据经过了某种预处理（如预加重、滤波等）
2. dat文件是从其他来源生成的原始数据
3. 两个文件的生成过程不同

当前最佳转换代码：
""")

# 输出最终的转换代码
print("```python")
print("import numpy as np")
print("")
print("# H2转换 - 使用线性映射")
print("npy_h2 = np.load('volRendering_H2.npy')")
print("converted = (npy_h2 + 1) / 2 * 255")
print("converted = converted.astype(np.float32)")
print("converted.tofile('H2_converted.dat')")
print("")
print("# PD转换 - 使用范围归一化")
print("npy_pd = np.load('volRendering_PD.npy')")
print("converted = (npy_pd - npy_pd.min()) / (npy_pd.max() - npy_pd.min()) * 255")
print("converted = converted.astype(np.float32)")
print("converted.tofile('PD_converted.dat')")
print("```")

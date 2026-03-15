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
print("H2 精确转换")
print("="*60)

# 方法：dat = round((npy + 1) * 127)
# 这样-1会变成0，1会变成254（接近255）
converted = np.round((npy_h2 + 1) * 127).astype(np.float32)

print(f"\n方法: round((npy + 1) * 127)")
print(f"转换后范围: [{converted.min():.4f}, {converted.max():.4f}]")
diff = np.abs(converted - dat_h2)
print(f"完全匹配: {np.sum(diff == 0)}/{len(diff)}")
print(f"最大差异: {diff.max():.4f}")

# 方法2：检查是否dat中的255对应npy中的1
# 尝试：dat = round((npy + 1) * 127.5) - 1
converted2 = np.round((npy_h2 + 1) * 127.5) - 1
converted2 = converted2.astype(np.float32)

print(f"\n方法2: round((npy + 1) * 127.5) - 1")
print(f"转换后范围: [{converted2.min():.4f}, {converted2.max():.4f}]")
diff2 = np.abs(converted2 - dat_h2)
print(f"完全匹配: {np.sum(diff2 == 0)}/{len(diff2)}")
print(f"最大差异: {diff2.max():.4f}")

# 方法3：dat = (npy + 1) * 127 + 0.5 然后取整
converted3 = np.floor((npy_h2 + 1) * 127 + 0.5).astype(np.float32)

print(f"\n方法3: floor((npy + 1) * 127 + 0.5)")
print(f"转换后范围: [{converted3.min():.4f}, {converted3.max():.4f}]")
diff3 = np.abs(converted3 - dat_h2)
print(f"完全匹配: {np.sum(diff3 == 0)}/{len(diff3)}")
print(f"最大差异: {diff3.max():.4f}")

# 方法4：既然npy中-1对应dat中0，npy中1应该对应dat中255
# 线性插值：dat = ((npy - (-1)) / (1 - (-1))) * 255
#       = ((npy + 1) / 2) * 255
converted4 = (npy_h2 + 1) / 2 * 255
converted4 = converted4.astype(np.float32)

print(f"\n方法4: (npy + 1) / 2 * 255")
print(f"转换后范围: [{converted4.min():.4f}, {converted4.max():.4f}]")
diff4 = np.abs(converted4 - dat_h2)
print(f"完全匹配: {np.sum(diff4 == 0)}/{len(diff4)}")
print(f"最大差异: {diff4.max():.4f}")

# 关键突破：让我仔细看dat中的值分布
print("\n" + "="*60)
print("深入分析差异")
print("="*60)

# 差异值为1的位置
diff_pos = np.where(diff == 1)[0]
print(f"差异=1的位置数: {len(diff_pos)}")

if len(diff_pos) > 0:
    # 这些位置的npy值和dat值
    npy_at_diff = npy_h2[diff_pos]
    dat_at_diff = dat_h2[diff_pos]
    conv_at_diff = converted[diff_pos]
    
    print(f"\n这些位置的分析:")
    print(f"  npy值范围: [{npy_at_diff.min():.6f}, {npy_at_diff.max():.6f}]")
    print(f"  dat值范围: [{dat_at_diff.min():.6f}, {dat_at_diff.max():.6f}]")
    print(f"  转换值范围: [{conv_at_diff.min():.6f}, {conv_at_diff.max():.6f}]")
    
    # 看看转换后比dat大还是小
    greater = np.sum(conv_at_diff > dat_at_diff)
    less = np.sum(conv_at_diff < dat_at_diff)
    print(f"  转换值>dat: {greater}, 转换值<dat: {less}")
    
    # 如果转换值比dat大1，说明需要减1
    # 尝试：对于(npy+1)*127.5 - dat = 1的情况，转换值应该-1
    
# 方法5：修正转换
# 问题是：(npy+1)*127.5 可能产生 0.5的情况
# 让我检查
print("\n" + "="*60)
print("方法5：逐元素修正")
print("="*60)

converted5 = (npy_h2 + 1) * 127.5
# 找出需要修正的位置
# 如果转换后值 - dat = 1，说明需要向下取整
# 如果dat - 转换后值 = 1，说明需要向上取整

# 简单做法：直接四舍五入
converted5_rounded = np.round(converted5).astype(np.float32)
diff5 = np.abs(converted5_rounded - dat_h2)
print(f"round((npy+1)*127.5) 匹配: {np.sum(diff5 == 0)}/{len(diff5)}")

# 检查剩余差异
remaining_diff = diff5[diff5 > 0]
print(f"剩余差异值: {np.unique(remaining_diff)[:10]}")

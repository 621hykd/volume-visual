import numpy as np
import os

# 定义文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
volume_dir = os.path.join(base_dir, "volume")

# 读取数据
npy_h2 = np.load(os.path.join(base_dir, "volRendering_H2.npy"))
dat_h2 = np.fromfile(os.path.join(volume_dir, "H2.dat"), dtype=np.float32)

print("="*60)
print("分析 H2 数据的归一化关系")
print("="*60)

print(f"\nnpy源数据:")
print(f"  范围: [{npy_h2.min():.6f}, {npy_h2.max():.6f}]")
print(f"  均值: {npy_h2.mean():.6f}")
print(f"  标准差: {npy_h2.std():.6f}")

print(f"\ndat目标数据:")
print(f"  范围: [{dat_h2.min():.6f}, {dat_h2.max():.6f}]")
print(f"  均值: {dat_h2.mean():.6f}")
print(f"  标准差: {dat_h2.std():.6f}")

# 尝试找出归一化参数
# 假设关系是: dat = a * npy + b
# 使用最小二乘法求解

# 方法1：线性回归
a = np.cov(npy_h2, dat_h2)[0,1] / np.var(npy_h2)
b = dat_h2.mean() - a * npy_h2.mean()

print(f"\n线性关系: dat = {a:.6f} * npy + {b:.6f}")

# 应用这个关系
converted = a * npy_h2 + b
converted = converted.astype(np.float32)

print(f"\n使用线性关系转换后:")
print(f"  范围: [{converted.min():.6f}, {converted.max():.6f}]")
print(f"  均值: {converted.mean():.6f}")

# 对比
diff = np.abs(converted - dat_h2)
print(f"\n对比结果:")
print(f"  最大差异: {diff.max():.6f}")
print(f"  平均差异: {diff.mean():.6f}")
print(f"  完全匹配: {np.sum(diff == 0)}/{len(diff)}")

if np.allclose(converted, dat_h2, rtol=1e-5):
    print("\n✓ 使用线性关系转换后完全一致！")
else:
    # 尝试另一种方法：分段线性映射
    print("\n尝试分段线性映射...")
    
    # 假设npy数据是[-1,1]但dat有偏移
    # 尝试找到最优的a和b
    from scipy import optimize
    
    def objective(params):
        a, b = params
        pred = a * npy_h2 + b
        return np.sum((pred - dat_h2)**2)
    
    result = optimize.minimize(objective, [127.5, 127.5], method='Nelder-Mead')
    a_opt, b_opt = result.x
    
    print(f"优化后的参数: a={a_opt:.6f}, b={b_opt:.6f}")
    
    converted_opt = a_opt * npy_h2 + b_opt
    converted_opt = converted_opt.astype(np.float32)
    
    diff_opt = np.abs(converted_opt - dat_h2)
    print(f"\n优化后对比:")
    print(f"  最大差异: {diff_opt.max():.6f}")
    print(f"  平均差异: {diff_opt.mean():.6f}")
    print(f"  完全匹配: {np.sum(diff_opt < 1e-5)}/{len(diff_opt)}")
    
    # 检查是否四舍五入后一致
    rounded = np.round(converted_opt)
    match_rounded = np.sum(rounded == dat_h2)
    print(f"  四舍五入后匹配: {match_rounded}/{len(rounded)}")

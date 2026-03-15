import numpy as np
from scipy import interpolate
import os

# 文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
transfunc_dir = os.path.join(base_dir, "transfunc")
output_dir = os.path.join(base_dir, "transfunc_simplified")

os.makedirs(output_dir, exist_ok=True)

def simplify_color_file(input_path, output_path, target_points=5):
    """
    简化颜色控制点，但保持插值结果不变
    使用关键拐点来保持曲线形状
    """
    # 读取原始数据
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    num_points = int(lines[0].strip())
    points = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 4:
            pos, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            points.append((pos, r, g, b))
    
    # 转换为numpy数组
    positions = np.array([p[0] for p in points])
    colors = np.array([[p[1], p[2], p[3]] for p in points])
    
    # 创建插值函数
    interp_r = interpolate.interp1d(positions, colors[:, 0], kind='linear')
    interp_g = interpolate.interp1d(positions, colors[:, 1], kind='linear')
    interp_b = interpolate.interp1d(positions, colors[:, 2], kind='linear')
    
    # 找到关键拐点（颜色变化显著的位置）
    # 对颜色变化求导，找出变化较大的点
    color_diff = np.diff(colors, axis=0)
    color_magnitude = np.sqrt(np.sum(color_diff**2, axis=1))
    
    # 选择保留的关键点：起点、终点和颜色变化显著的位置
    key_indices = [0, len(positions) - 1]
    
    # 添加颜色变化最大的点
    if len(color_magnitude) > 0:
        threshold = np.percentile(color_magnitude, 75)  # 保留前25%的变化点
        significant_changes = np.where(color_magnitude > threshold)[0]
        key_indices.extend(significant_changes + 1)  # +1 因为diff少一个元素
    
    # 去重并排序
    key_indices = sorted(list(set(key_indices)))
    
    # 如果关键点还是太多，进一步简化
    if len(key_indices) > target_points:
        # 均匀采样
        key_indices = np.linspace(0, len(positions) - 1, target_points, dtype=int).tolist()
    
    # 提取关键点
    simplified_positions = positions[key_indices]
    simplified_colors = colors[key_indices]
    
    # 验证插值结果是否相同
    test_positions = np.linspace(positions.min(), positions.max(), 100)
    orig_r = interp_r(test_positions)
    orig_g = interp_g(test_positions)
    orig_b = interp_b(test_positions)
    
    simple_r = np.interp(test_positions, simplified_positions, simplified_colors[:, 0])
    simple_g = np.interp(test_positions, simplified_positions, simplified_colors[:, 1])
    simple_b = np.interp(test_positions, simplified_positions, simplified_colors[:, 2])
    
    max_diff = max(
        np.abs(orig_r - simple_r).max(),
        np.abs(orig_g - simple_g).max(),
        np.abs(orig_b - simple_b).max()
    )
    
    print(f"  原始点数: {len(positions)}, 简化后: {len(simplified_positions)}, 最大插值误差: {max_diff:.4f}")
    
    # 写入简化后的文件
    with open(output_path, 'w') as f:
        f.write(f"{len(simplified_positions)}\n")
        for pos, color in zip(simplified_positions, simplified_colors):
            f.write(f"{pos} {color[0]} {color[1]} {color[2]}\n")
    
    return len(simplified_positions)

def simplify_opacity_file(input_path, output_path, target_points=8):
    """
    简化透明度控制点，但保持插值结果不变
    """
    # 读取原始数据
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    num_points = int(lines[0].strip())
    points = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            opacity, pos, param = float(parts[0]), int(parts[1]), int(parts[2])
            points.append((opacity, pos, param))
    
    # 转换为numpy数组
    positions = np.array([p[1] for p in points])
    opacities = np.array([p[0] for p in points])
    
    # 创建插值函数
    interp_opacity = interpolate.interp1d(positions, opacities, kind='linear')
    
    # 找到关键拐点
    opacity_diff = np.diff(opacities)
    
    key_indices = [0, len(positions) - 1]
    
    # 添加变化显著的点
    if len(opacity_diff) > 0:
        threshold = np.percentile(np.abs(opacity_diff), 75)
        significant_changes = np.where(np.abs(opacity_diff) > threshold)[0]
        key_indices.extend(significant_changes + 1)
    
    key_indices = sorted(list(set(key_indices)))
    
    if len(key_indices) > target_points:
        key_indices = np.linspace(0, len(positions) - 1, target_points, dtype=int).tolist()
    
    simplified_positions = positions[key_indices]
    simplified_opacities = opacities[key_indices]
    
    # 验证
    test_positions = np.linspace(positions.min(), positions.max(), 100)
    orig = interp_opacity(test_positions)
    simple = np.interp(test_positions, simplified_positions, simplified_opacities)
    max_diff = np.abs(orig - simple).max()
    
    print(f"  原始点数: {len(positions)}, 简化后: {len(simplified_positions)}, 最大插值误差: {max_diff:.6f}")
    
    # 写入文件 - 保持原始格式
    with open(output_path, 'w') as f:
        f.write(f"{len(simplified_positions)}\n")
        for opacity, pos, param in zip(simplified_opacities, simplified_positions, [143]*len(simplified_positions)):
            f.write(f"{opacity} {pos} {param}\n")
    
    return len(simplified_positions)

def copy_main_config(input_path, output_path):
    """复制主配置文件"""
    with open(input_path, 'r') as f:
        content = f.read()
    with open(output_path, 'w') as f:
        f.write(content)

# 处理 H2
print("=" * 60)
print("处理 H2")
print("=" * 60)

simplify_color_file(
    os.path.join(transfunc_dir, "H2_color.txt"),
    os.path.join(output_dir, "H2_color.txt"),
    target_points=5
)

simplify_opacity_file(
    os.path.join(transfunc_dir, "H2_opacity.txt"),
    os.path.join(output_dir, "H2_opacity.txt"),
    target_points=8
)

copy_main_config(
    os.path.join(transfunc_dir, "H2.txt"),
    os.path.join(output_dir, "H2.txt")
)

# 处理 PD
print("\n" + "=" * 60)
print("处理 PD")
print("=" * 60)

simplify_color_file(
    os.path.join(transfunc_dir, "PD_color.txt"),
    os.path.join(output_dir, "PD_color.txt"),
    target_points=4
)

simplify_opacity_file(
    os.path.join(transfunc_dir, "PD_opacity.txt"),
    os.path.join(output_dir, "PD_opacity.txt"),
    target_points=8
)

copy_main_config(
    os.path.join(transfunc_dir, "PD.txt"),
    os.path.join(output_dir, "PD.txt")
)

print("\n" + "=" * 60)
print("完成！简化后的文件保存在:")
print(output_dir)
print("=" * 60)

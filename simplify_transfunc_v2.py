import numpy as np
import os

# 文件路径
base_dir = r"C:\Users\30114\Desktop\volume-visual"
transfunc_dir = os.path.join(base_dir, "transfunc")
output_dir = os.path.join(base_dir, "transfunc_simplified")

os.makedirs(output_dir, exist_ok=True)

def simplify_color_file_precise(input_path, output_path):
    """
    精确简化颜色控制点
    策略：保留所有拐点（颜色方向变化的点）
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    num_points = int(lines[0].strip())
    points = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 4:
            pos, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            points.append((pos, r, g, b))
    
    if len(points) <= 3:
        # 点太少，直接复制
        with open(output_path, 'w') as f:
            f.write(f"{len(points)}\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
        return len(points)
    
    # 分析颜色变化方向，找出拐点
    # 拐点定义：颜色变化方向发生显著变化的点
    
    colors = np.array([[p[1], p[2], p[3]] for p in points])
    positions = np.array([p[0] for p in points])
    
    # 计算相邻颜色之间的差异向量
    diffs = np.diff(colors, axis=0)
    
    # 计算相邻差异向量之间的角度变化
    key_indices = [0]  # 始终保留起点
    
    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        
        # 计算角度变化
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_diff = np.arccos(cos_angle)
            
            # 如果角度变化超过阈值，认为是拐点
            if angle_diff > 0.3:  # 约17度
                key_indices.append(i + 1)
    
    key_indices.append(len(points) - 1)  # 保留终点
    
    # 去重并排序
    key_indices = sorted(list(set(key_indices)))
    
    # 提取关键点
    simplified = [points[i] for i in key_indices]
    
    # 验证插值结果
    test_pos = np.linspace(positions[0], positions[-1], 200)
    
    # 原始插值
    orig_r = np.interp(test_pos, positions, colors[:, 0])
    orig_g = np.interp(test_pos, positions, colors[:, 1])
    orig_b = np.interp(test_pos, positions, colors[:, 2])
    
    # 简化后插值
    simple_pos = np.array([points[i][0] for i in key_indices])
    simple_colors = np.array([[p[1], p[2], p[3]] for p in simplified])
    
    simple_r = np.interp(test_pos, simple_pos, simple_colors[:, 0])
    simple_g = np.interp(test_pos, simple_pos, simple_colors[:, 1])
    simple_b = np.interp(test_pos, simple_pos, simple_colors[:, 2])
    
    max_diff = max(
        np.abs(orig_r - simple_r).max(),
        np.abs(orig_g - simple_g).max(),
        np.abs(orig_b - simple_b).max()
    )
    
    print(f"  颜色: 原始{len(points)}点 -> 简化{len(simplified)}点, 最大误差: {max_diff:.2f}")
    
    # 如果误差太大，添加更多点
    if max_diff > 1:
        # 使用更保守的方法：保留所有点
        simplified = points
        max_diff = 0
    
    # 写入
    with open(output_path, 'w') as f:
        f.write(f"{len(simplified)}\n")
        for p in simplified:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
    
    return len(simplified)

def simplify_opacity_file_precise(input_path, output_path):
    """
    精确简化透明度控制点
    策略：保留所有斜率变化的点
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    num_points = int(lines[0].strip())
    points = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            opacity, pos, param = float(parts[0]), int(parts[1]), int(parts[2])
            points.append((opacity, pos, param))
    
    if len(points) <= 3:
        with open(output_path, 'w') as f:
            f.write(f"{len(points)}\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        return len(points)
    
    positions = np.array([p[1] for p in points])
    opacities = np.array([p[0] for p in points])
    params = np.array([p[2] for p in points])
    
    # 找出斜率变化的点
    key_indices = [0]
    
    # 计算斜率
    for i in range(len(positions) - 1):
        if positions[i+1] != positions[i]:
            slope = (opacities[i+1] - opacities[i]) / (positions[i+1] - positions[i])
            
            # 检查斜率变化
            if i > 0 and i < len(positions) - 2:
                prev_slope = (opacities[i] - opacities[i-1]) / (positions[i] - positions[i-1])
                next_slope = (opacities[i+2] - opacities[i+1]) / (positions[i+2] - positions[i+1])
                
                # 如果斜率变化超过阈值
                if abs(slope - prev_slope) > 0.001 or abs(slope - next_slope) > 0.001:
                    if i not in key_indices:
                        key_indices.append(i)
    
    key_indices.append(len(points) - 1)
    key_indices = sorted(list(set(key_indices)))
    
    simplified = [points[i] for i in key_indices]
    
    # 验证
    test_pos = np.linspace(positions[0], positions[-1], 200)
    orig = np.interp(test_pos, positions, opacities)
    
    simple_pos = np.array([points[i][1] for i in key_indices])
    simple_opa = np.array([points[i][0] for i in key_indices])
    simple = np.interp(test_pos, simple_pos, simple_opa)
    
    max_diff = np.abs(orig - simple).max()
    print(f"  透明度: 原始{len(points)}点 -> 简化{len(simplified)}点, 最大误差: {max_diff:.6f}")
    
    # 写入
    with open(output_path, 'w') as f:
        f.write(f"{len(simplified)}\n")
        for p in simplified:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    
    return len(simplified)

def copy_main_config(input_path, output_path):
    with open(input_path, 'r') as f:
        content = f.read()
    with open(output_path, 'w') as f:
        f.write(content)

# 处理 H2
print("=" * 60)
print("处理 H2")
print("=" * 60)

simplify_color_file_precise(
    os.path.join(transfunc_dir, "H2_color.txt"),
    os.path.join(output_dir, "H2_color.txt")
)

simplify_opacity_file_precise(
    os.path.join(transfunc_dir, "H2_opacity.txt"),
    os.path.join(output_dir, "H2_opacity.txt")
)

copy_main_config(
    os.path.join(transfunc_dir, "H2.txt"),
    os.path.join(output_dir, "H2.txt")
)

# 处理 PD
print("\n" + "=" * 60)
print("处理 PD")
print("=" * 60)

simplify_color_file_precise(
    os.path.join(transfunc_dir, "PD_color.txt"),
    os.path.join(output_dir, "PD_color.txt")
)

simplify_opacity_file_precise(
    os.path.join(transfunc_dir, "PD_opacity.txt"),
    os.path.join(output_dir, "PD_opacity.txt")
)

copy_main_config(
    os.path.join(transfunc_dir, "PD.txt"),
    os.path.join(output_dir, "PD.txt")
)

print("\n完成！")

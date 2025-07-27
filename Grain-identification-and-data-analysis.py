import cv2
import numpy as np
import time
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
import pandas as pd  # 添加pandas库用于数据处理
import math  # 用于计算等效直径


# 预处理图像
def yuchuli(image, threshold):
    print(f"-----图像处理开始-----")
    start_time = time.time()

    # 直接处理为灰度图，跳过不必要的中间保存
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"图像尺寸: {gray_image.shape}")

    # CLAHE提升对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # 增加tile尺寸加速
    clahe_image = clahe.apply(gray_image)

    # 二值化
    _, threshold_image = cv2.threshold(clahe_image, threshold, 255, cv2.THRESH_BINARY)

    #threshold_image[:, :50] = 0  # 修改前left列为黑色
    #threshold_image[:, :left-1] = 255  # 修改前left列为白色
    print(f"预处理总时间: {time.time() - start_time:.3f}s")
    return threshold_image


# 噪声去除
def remove_noises(binary_img, min_white_area=25, min_black_area=25):
    start_time = time.time()

    # 去除白色噪点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=4
    )
    small_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] < min_white_area]
    if small_labels:
        mask = np.isin(labels, small_labels)
        binary_img[mask] = 0

    # 去除黑色噪点（反转处理）
    inverted = 255 - binary_img
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=4
    )
    small_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] < min_black_area]
    if small_labels:
        mask = np.isin(labels, small_labels)
        binary_img[mask] = 255

    print(f"噪声去除时间: {time.time() - start_time:.3f}s")
    return binary_img


# 分块处理函数
def process_large_image(image_path, block_width=1436, overlap=200):
    print(f"开始处理大图像分块...")
    full_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if full_image is None:
        print(f"无法读取图像: {image_path}")
        return None

    height, width = full_image.shape[:2]
    print(f"原始图像尺寸: {height}×{width}")

    # 创建空的结果数组
    result_image = np.zeros((height, width), dtype=np.uint8)

    # 计算分块数量
    stride = block_width - overlap  # 有效前进量
    num_blocks = (width - overlap + stride - 1) // stride  # 向上取整
    print(f"将分为 {num_blocks} 个分块 (每块宽度: {block_width}, 重叠: {overlap}, 步长: {stride})")

    # 实际开始位置
    current_pos = 0

    for i in tqdm(range(num_blocks), desc="处理分块"):
        # 计算当前块起始和结束位置
        start_x = i * stride
        end_x = min(start_x + block_width, width)

        # 当前块实际宽度
        actual_width = end_x - start_x
        print(f"\n处理块 {i + 1}/{num_blocks}: 位置[{start_x}-{end_x}], 宽度={actual_width}")

        # 提取当前块
        block = full_image[:, start_x:end_x]

        # 处理当前块
        processed_block = process_image_block(block)

        # 确定要复制的内容范围
        if i == 0:  # 第一块: 保留整块减去右侧重叠
            content_start = 0
            content_end = min(actual_width - overlap // 2, actual_width)
        elif i == num_blocks - 1:  # 最后一块: 保留整块减去左侧重叠
            content_start = overlap // 2
            content_end = actual_width
        else:  # 中间块: 只保留中心部分，去重
            content_start = overlap // 2
            content_end = min(actual_width - overlap // 2, actual_width)

        content_width = content_end - content_start
        print(f"复制范围: [{content_start}-{content_end}], 宽度={content_width}")

        # 复制到结果图像
        result_end = min(current_pos + content_width, width)
        result_slice = slice(current_pos, result_end)

        # 确保尺寸匹配
        if processed_block.shape[1] < content_width:
            padding = np.zeros((height, content_width - processed_block.shape[1]), dtype=np.uint8)
            processed_block = np.hstack((processed_block, padding))

        result_image[:, result_slice] = processed_block[:, content_start:content_end]

        # 更新位置指针
        current_pos = result_end

    print(f"分块处理完成! 结果宽度={current_pos}")
    return result_image


# 处理单个图像块
def process_image_block(block):
    # 1. 预处理
    binary = yuchuli(block, threshold=160)

    # 2. 去噪
    denoised = remove_noises(binary, min_white_area=0, min_black_area=10)

    # 3. 连接断点
    # connected = connect_line_gaps(denoised, max_gap=3)

    # 4. 区域合并
    merged = merge_small_white_regions(denoised, region_size=256,  # 区域合并搜索框
                                       black_threshold=0.4,        # 黑色占比阈值(0~1)
                                       max_white_area=10,         # 合并白色区域最大值
                                       min_area_large=10,        # 排除大区域最小值
                                       shengzhanghe=1)
    merged = merge_small_white_regions(merged, 256, 0.3, 20, 20, 5)
    merged = merge_small_white_regions(merged, 256, 0.2, 50, 50, 5)
    merged = merge_small_white_regions(merged, 256, 0.2, 150, 150, 10)
    merged = merge_small_white_regions(merged, 256, 0.2, 250, 250, 10)
    merged = merge_small_white_regions(merged, 256, 0.2, 350, 350, 10)
    # 4. 去噪
    merged = remove_noises(merged, min_white_area=0, min_black_area=30)

    return merged


# 连接断点函数（目前弃用了）
def connect_line_gaps(image, max_gap):
    start_connect_time = time.time()
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    output = np.zeros_like(image)
    cv2.drawContours(output, contours, -1, 255, 1)

    # 收集所有端点并建立KD树
    endpoints = []
    for cnt in contours:
        if len(cnt) > 1:
            endpoints.append(cnt[0][0])
            endpoints.append(cnt[-1][0])

    if not endpoints:
        return 255 - output

    # 使用KDTree加速邻近点查找
    endpoints_arr = np.array(endpoints)
    kd_tree = cKDTree(endpoints_arr)

    # 批量查询邻近点对
    pairs = kd_tree.query_pairs(max_gap)

    # 连接符合距离条件的端点
    for i, j in pairs:
        pt1 = tuple(endpoints_arr[i])
        pt2 = tuple(endpoints_arr[j])
        distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        if 1 < distance < max_gap:
            cv2.line(output, pt1, pt2, 255, 1)

    connect_line_time = time.time()
    cv2.imwrite("connect_line.jpg", 255 - output)
    print(f"线性连接处理时间: {connect_line_time - start_connect_time:.3f}s")
    print(f"-----图像处理结束-----")
    return 255 - output


# 区域合并函数
def merge_small_white_regions(binary_img, region_size, black_threshold, max_white_area, min_area_large, shengzhanghe):
    """
    带外部边界保护的区域合并
    （先生成面积大于min_area_large的晶粒的黑色外晶界作备用，
    再融合所有小于max_white_area的白色晶粒，
    最后把黑色外晶界放回）
    参数:
        binary_img: 二值图像(0=黑, 255=白)
        region_size: 检测区域大小(正方形边长)
        black_threshold: 黑色占比阈值(0~1)
        max_white_area: 合并白色区域最大值
        min_area_large: 排除大区域最小值
    """
    print(f"-----带外部边界保护的区域合并开始-----")
    start_time = time.time()
    height, width = binary_img.shape

    # 1. 创建原始图像副本（图1）
    original_img = binary_img.copy()

    # 2. 创建保护边界掩膜（图2）
    boundary_mask = np.zeros((height, width), dtype=np.uint8)

    # 全局连通域分析用于识别大区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=4
    )

    # 创建只包含大区域的掩膜
    large_regions_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_large:
            large_regions_mask[labels == i] = 255

    # 为每个大区域创建外部边界（在大区域外绘制黑色边界）
    large_regions_mask = cv2.medianBlur(large_regions_mask, 3)  # 将大晶粒的边界平滑处理
    kernel = np.ones((3, 3), np.uint8)
    dilated_regions = cv2.dilate(large_regions_mask, kernel, iterations=2) # iterations=1那么黑色晶界就是几
    external_boundary = dilated_regions - large_regions_mask
    boundary_mask = external_boundary.astype(np.uint8)

    # 保存中间结果（调试）

    cv2.imwrite("large_regions_mask.jpg", large_regions_mask)
    cv2.imwrite("dilated_regions.jpg", dilated_regions)
    cv2.imwrite("external_boundary.jpg", boundary_mask)

    # 3. 在原始图像上执行小白块合并操作（图1 -> 图3）
    # 滑动窗口检测黑色密集区域
    region_mask = np.zeros((height, width), dtype=np.uint8)
    half_size = region_size // 2
    step = max(1, region_size // 3)

    for y in range(0, height, step):
        for x in range(0, width, step):
            y1 = max(0, y - half_size)
            y2 = min(height, y + half_size + 1)
            x1 = max(0, x - half_size)
            x2 = min(width, x + half_size + 1)

            region = binary_img[y1:y2, x1:x2]
            black_pixels = np.sum(region == 0)
            total_pixels = region.size

            if black_pixels / total_pixels > black_threshold:
                region_mask[y1:y2, x1:x2] = 255

    # 执行小白块合并
    merged_img = original_img.copy()

    # 对小连通域进行分析
    small_components = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < max_white_area:
            small_components.append(i)

    if small_components:
        # 创建小区域合并掩膜
        small_mask = np.zeros((height, width), dtype=np.uint8)
        for i in small_components:
            small_mask[labels == i] = 255

        # 只合并位于黑色密集区域的小白块
        small_mask = cv2.bitwise_and(small_mask, region_mask)

        # 合并处理：将小白块区域扩张并设置为白色
        merged_mask = cv2.dilate(small_mask, kernel, iterations = shengzhanghe)
        merged_img[merged_mask == 255] = 255
        cv2.imwrite("small_mask.jpg", small_mask)
        cv2.imwrite("merged_mask.jpg", merged_mask)

    # 4. 将保护边界应用到合并后的图像（图3 -> 图4）
    final_output = merged_img.copy()
    final_output[boundary_mask == 255] = 0  # 在边界位置设置为黑色

    # 后处理优化：平滑处理
    # final_output = cv2.medianBlur(final_output, 3)

    # 统计信息
    large_region_count = sum(1 for i in range(1, num_labels)
                             if stats[i, cv2.CC_STAT_AREA] >= min_area_large)
    small_region_count = len(small_components)

    print(f"检测到大区域数量: {large_region_count}")
    print(f"检测到小区域数量: {small_region_count}")
    print(f"处理区域尺寸: {region_size}×{region_size}")
    print(f"黑色像素阈值: {black_threshold * 100:.0f}%")
    print(f"合并白色区域阈值: <{max_white_area}像素")
    print(f"处理时间: {time.time() - start_time:.3f}s")
    print(f"-----小区域白色合并处理结束-----")

    cv2.imwrite("merge_small_white_external_boundary.jpg", final_output)
    return final_output


# 记录晶粒位置和等效直径
def record_grain_data(stats, centroids, min_area, output_csv="grain_data.csv"):
    """
    记录晶粒的位置和等效直径
    参数:
        stats: 连通域统计信息
        centroids: 连通域质心坐标
        min_area: 最小晶粒面积阈值
    返回:
        DataFrame: 包含晶粒信息的表格
    """
    grain_data = []

    # 遍历所有连通域(跳过背景0)
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # 计算等效直径: 2 * √(area/π)
        equivalent_diameter = 2 * math.sqrt(area* 2500 / 137 / 137 / math.pi)

        # 获取x坐标(质心的x位置)
        x_coord = int(centroids[i][0])

        grain_data.append({
            "ID": i,
            "X_Coordinate": x_coord,
            "Area": area,
            "Equivalent_Diameter": equivalent_diameter
        })

    # 创建DataFrame并保存
    df = pd.DataFrame(grain_data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"已保存晶粒数据到 {output_csv}，共 {len(df)} 个晶粒")
    else:
        print("未找到符合条件的晶粒")

    return df


# 晶粒识别函数
def bianyuan(image, left, kernel_min, area_min, block_width=1436, overlap=200):
    # 确保传入的图像是3通道BGR格式
    if len(image.shape) == 2:  # 如果是灰度图像
        # 转换为BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:  # 如果是单通道图像
        # 转换为3通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #image[:, :left] = 0  # 修改前left列为黑色
    #image[:, :left-1] = 255  # 修改前left列为白色

    height, width = image.shape[:2]
    print(f"-----晶粒识别开始 (图像尺寸: {height}×{width})-----")

    # 创建结果图像
    result_image = np.zeros_like(image, dtype=np.uint8)

    # 计算分块数量
    stride = block_width - overlap  # 有效前进量
    num_blocks = (width - overlap + stride - 1) // stride  # 向上取整
    print(f"将分为 {num_blocks} 个分块进行晶粒识别 (每块宽度: {block_width}, 重叠: {overlap}, 步长: {stride})")

    # 存储所有晶粒的面积信息
    all_area_stats = []
    total_grains = 0
    removed_count = 0
    #removed_area = 0

    # 创建全局颜色映射表
    global_color_map = {}

    # 全局存储晶粒信息
    all_grain_stats = []
    all_centroids = []
    next_id = 1  # 晶粒ID从1开始

    # 分块处理
    for i in tqdm(range(num_blocks), desc="晶粒识别分块"):
        # 计算当前块起始和结束位置
        start_x = i * stride
        end_x = min(start_x + block_width, width)

        # 当前块实际宽度
        actual_width = end_x - start_x

        # 提取当前块
        block = image[:, start_x:end_x].copy()

        # 为当前块分配晶粒ID范围
        block_ids_start = next_id

        # 处理当前块
        filled_block, area_stats, block_grains, block_removed, block_stats, block_centroids = process_block_grains(
            block, kernel_min, area_min, global_color_map, start_id=next_id
        )

        # 收集晶粒信息并更新ID
        all_grain_stats.extend(block_stats)
        all_centroids.extend(block_centroids)
        next_id += block_grains  # 更新ID计数器

        # 记录晶粒位置（调整全局坐标）
        for j, centroid in enumerate(block_centroids):
            if block_centroids[j][0] != -1:  # 有效质心
                block_centroids[j][0] += start_x  # 调整x坐标为全局位置

        # 复制到结果图像
        result_image[:, start_x:end_x] = filled_block

        # 更新统计信息
        all_area_stats.extend(area_stats)
        total_grains += block_grains
        removed_count += block_removed

    # 保存结果
    cv2.imwrite("filled_regions.jpg", result_image)

    # 合并全局晶粒信息
    if all_grain_stats and all_centroids:
        # 转换为NumPy数组
        stats_array = np.vstack(all_grain_stats)
        centroids_array = np.vstack(all_centroids)

        # 记录晶粒数据
        record_grain_data(stats_array, centroids_array, area_min)

    # 分析区域面积分布
    print(f"检测到 {total_grains} 个有效晶粒")
    print(f"移除 {removed_count} 个小区域")
    print(f"最小晶粒面积: {min(all_area_stats) if all_area_stats else 0}")
    print(f"最大晶粒面积: {max(all_area_stats) if all_area_stats else 0}")
    print(f"平均晶粒面积: {sum(all_area_stats) / len(all_area_stats) if all_area_stats else 0}")
    print(f"-----晶粒识别结束-----")

    return result_image


# 单个块的晶粒识别
def process_block_grains(block, kernel_min, area_min, color_map=None, start_id=1):
    """
    处理单个块的晶粒识别
    返回添加了晶粒统计数据
    """
    # 转为灰度图
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)

    # 增强对比度（解决模糊边界问题）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # 二值化
    _, binary = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学处理：断开细小的连接
    #kernel = np.ones((kernel_min, kernel_min), np.uint8)
    #processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1) #断开小于kernel_min的小缝隙iterations次

    # 使用连通组件分析（带统计信息）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)

    # 创建彩色填充图像
    filled = np.zeros_like(block, dtype=np.uint8)

    # 存储面积信息用于分析
    area_stats = []

    # 记录被剔除的连通域数量和面积
    block_removed = 0

    # 存储本块晶粒统计信息
    block_stats = []
    block_centroids = []

    # 当前晶粒ID
    current_id = start_id

    # 遍历所有连通区域（跳过背景0）
    for label in range(1, num_labels):
        # 获取当前连通区域的统计信息
        x, y, w, h, area = stats[label]
        area_stats.append(area)

        # 无效质心占位符
        centroid = [-1, -1]

        if area < area_min:  # 忽略较小的区域
            block_removed += 1
        else:
            # 添加晶粒统计信息
            block_stats.append(stats[label])
            centroid = centroids[label]

            # 使用全局颜色映射确保颜色一致性
            if color_map is not None:
                if current_id not in color_map:
                    # 为新晶粒生成随机颜色
                    color_map[current_id] = (
                        np.random.randint(0, 256),
                        np.random.randint(0, 256),
                        np.random.randint(0, 256)
                    )
                color = color_map[current_id]
            else:
                # 没有颜色映射时生成随机颜色
                color = (np.random.randint(0, 256),
                         np.random.randint(0, 256),
                         np.random.randint(0, 256))

            # 在填充图像中着色该区域
            filled[labels == label] = color
            current_id += 1

        block_centroids.append(centroid)

    return (
        filled,
        area_stats,
        num_labels - 1 - block_removed,  # 有效晶粒数
        block_removed,
        block_stats,
        block_centroids
    )


# 修复被截断晶粒的坐标
def fix_fragmented_grains(final_processed_image, csv_path="grain_data.csv", min_area=25):
    """
    修复因分块处理而被截断晶粒的坐标
    参数:
        final_processed_image: 整个处理后的二值图像
        csv_path: 包含晶粒数据的CSV文件路径
        min_area: 最小晶粒面积阈值
    """
    print("-----开始修复截断晶粒坐标-----")
    start_time = time.time()

    # 1. 读取现有晶粒数据
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"未找到文件 {csv_path}")
        return

    # 2. 在整个图像上执行连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        final_processed_image, connectivity=4
    )

    # 3. 找出所有有效晶粒（面积大于阈值）
    valid_grains = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y = centroids[i]
            equivalent_diameter = 2 * math.sqrt(area*2500 / 137 / 137 / math.pi)
            valid_grains.append({
                "ID": i,
                "X_Coordinate": int(x),
                "Area": area,
                "Equivalent_Diameter": equivalent_diameter
            })

    # 4. 创建KD树用于快速查找邻近晶粒
    valid_points = np.array([(g['X_Coordinate'], g['Area']) for g in valid_grains])
    kd_tree = cKDTree(valid_points[:, :1])  # 只使用X坐标建立树

    # 5. 修复无效坐标的晶粒
    fixed_count = 0
    for index, row in df.iterrows():
        if row['X_Coordinate'] == -1:
            # 使用面积作为匹配依据
            area = row['Area']

            # 查找面积最接近的有效晶粒
            distances = np.abs(valid_points[:, 1] - area)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < area * 0.99:  # 面积差异小于50%
                closest_grain = valid_grains[closest_idx]
                df.at[index, 'X_Coordinate'] = closest_grain['X_Coordinate']
                fixed_count += 1
                print(f"修复晶粒ID {row['ID']}: 面积={area} -> 新坐标={closest_grain['X_Coordinate']}")

    # 6. 保存修复后的数据
    df.to_csv(csv_path, index=False)
    print(f"修复了 {fixed_count} 个无效坐标的晶粒")
    print(f"修复截断晶粒时间: {time.time() - start_time:.3f}s")
    print("-----修复截断晶粒完成-----")
    return fixed_count


# 修改主程序调用部分
if __name__ == '__main__':
    image_path = 'hll/3-D3-1.JPG'

    # 分块处理大图像
    final_processed = process_large_image(image_path)
    cv2.imwrite("final_processed.jpg", final_processed)
    print("保存进行晶粒识别前的最终处理图像完成!")

    # 执行晶粒识别（同样分块处理）
    left = 50  # 目前闲置了，但不能注掉
    kernel_min = 3
    area_min = 25
    block_width = 1436  # 目前闲置了，但不能注掉
    # 传递灰度图像给bianyuan函数
    colored_image = bianyuan(final_processed, left, kernel_min, area_min, block_width)
    # 修复被截断晶粒的坐标
    fixed_count = fix_fragmented_grains(final_processed, min_area=area_min)

    if fixed_count == 0:
        print("未修复任何晶粒坐标，可能需要调整匹配阈值")

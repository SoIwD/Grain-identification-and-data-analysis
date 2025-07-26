# 技术流程文档
SoIwD 2025.07.27

本代码适用于处理高分辨率金相图像。主要针对暗区晶界不明显的图像有较好的效果。
系统采用分块处理策略，结合连通域算法，能够准确识别晶粒和提取数据（位置、面积、等效直径）。

# 核心功能

1. 大图像分块处理：解决内存限制问题
2. 图像预处理：增强对比度、二值化、去噪
3. 晶粒边界保护：保留大晶粒边界
4. 小区域合并：智能合并小晶粒区域
5. 晶粒识别与着色：识别晶粒并着色可视化
6. 数据分析：记录晶粒位置、面积和等效直径
7. 截断晶粒坐标修复：处理分块边界处的晶粒截断导致的坐标问题

# 主要参数说明

分块处理参数

• block_width：分块宽度（默认1436*1436像素）

• overlap：块间重叠区域（默认200像素）

• stride：有效前进量（block_width - overlap）

预处理参数yuchuli(image, threshold)

• threshold：二值化阈值（默认150，提高会导致图像变暗，优点是晶界更清晰，缺点是暗区更暗）

• clipLimit：CLAHE对比度限制（默认2.0）

• tileGridSize：CLAHE网格大小（默认16×16）

噪声去除参数remove_noises(binary_img, min_white_area=25, min_black_area=25)

• min_white_area：去除白色噪点面积（默认25像素）

• min_black_area：去除黑色噪点面积（默认25像素）

区域合并参数merge_small_white_regions(binary_img, region_size, black_threshold, max_white_area, min_area_large, shengzhanghe)

• region_size：检测区域大小（默认256像素，一个滑动的检测窗口）

• black_threshold：黑色占比阈值（0-1，若监测区域内的黑色占比超过阈值，则需要对该区域内的白色斑块进行“生长”操作）

• max_white_area：合并白色区域最大值（低于此值的白色斑块会生长）

• min_area_large：排除大区域最小值（高于此值的白色斑块不会参与生长且晶界不会被小白色斑块的“生长”所影响）

• shengzhanghe：膨胀迭代次数（值越大白色斑块成长越大）

晶粒识别参数bianyuan(image, left, kernel_min, area_min, block_width=1436, overlap=200)

• kernel_min：形态学操作核大小（默认3）

• area_min：最小晶粒面积（默认25像素）

• block_width：分块宽度（默认1436*1436像素）

• overlap：块间重叠区域（默认200像素）

# 使用流程

# 1. 图像预处理

image = cv2.imread('image.jpg')

processed = yuchuli(image, threshold=150)


# 2. 噪声去除

clean_image = remove_noises(processed, min_white_area=25, min_black_area=25)


# 3. 大图像分块处理

result = process_large_image('large_image.jpg', 
                            block_width=1436, 
                            overlap=200)


# 4. 区域合并（多次调用不同参数）

merged = merge_small_white_regions(image, 256, 0.4, 10, 10, 1)

merged = merge_small_white_regions(merged, 256, 0.3, 20, 20, 5)

...更多合并步骤...


# 5. 晶粒识别与数据分析

colored_image = bianyuan(processed_image, 
                        left=50, 
                        kernel_min=3, 
                        area_min=25,
                        block_width=1436)

自动生成grain_data.csv文件


# 6. 修复截断晶粒

fix_fragmented_grains(final_processed_image)

• 输出文件

(1). 预处理结果：final_processed.jpg

(2). 中间调试图像：

large_regions_mask.jpg（大区域掩膜）

dilated_regions.jpg（膨胀后区域）

external_boundary.jpg（外部边界）

small_mask.jpg（小区域掩膜）

merged_mask.jpg（合并掩膜）

(3). 晶粒着色图：filled_regions.jpg

(4). 晶粒数据：grain_data.csv（包含ID、X坐标、面积、等效直径）

• 性能优化

(1). 分块处理：处理超大图像（>10000×10000像素）

(2). KD树加速：邻近点快速查找

(3). 滑动窗口优化：步长控制处理效率

(4). 全局颜色映射：保持晶粒颜色一致性

(5). 增量ID分配：避免晶粒ID冲突

• 关键算法(区域合并算法)

(1). 创建大区域掩膜（面积>min_area_large）

(2). 生成外部边界（保护大晶粒边界）

(3). 滑动窗口检测黑色密集区域

(4). 合并小白色区域（面积<max_white_area）

(5). 应用保护边界

• 晶粒数据分析

(1). 等效直径计算：d = 2 \times \sqrt{\frac{area}{\pi}}

(2). 位置记录：晶体质心X坐标

(3). 面积统计：最小、最大、平均面积

• 使用建议

(1). 对于新图像，建议从默认参数开始

(2). 根据晶粒大小调整area_min参数

(3). 图像质量差时增加去噪强度

(4). 分块大小应根据内存容量调整

(5). 多次调用区域合并函数可获得更好效果


通过合理调整参数，本系统可适应各种金相图像的晶粒分析需求，提供准确的晶粒识别和量化数据。

P.S.搭配高质量的金相图使用效果更佳

# Technical Process Documentation
SoIwD 2025.07.27
[English](README.md) | [简体中文](README_zh.md)

<!-- Default language content -->
This code is suitable for processing high-resolution metallographic images. It is particularly effective for images with low-contrast grain boundaries in dark regions.
The system adopts a block-based processing strategy, combined with a connected component algorithm, enabling accurate grain identification and data extraction (position, area, equivalent diameter).

------------------------------------------------------------------------------------------------------------------------
# Core Features

1.  Large Image Block Processing: Solves memory limitation issues.
2.  Image Preprocessing: Enhances contrast, performs binarization, and removes noise.
3.  Grain Boundary Protection: Preserves boundaries of large grains.
4.  Small Region Merging: Intelligently merges small grain regions.
5.  Grain Identification and Coloring: Identifies grains and visualizes them with coloring.
6.  Data Analysis: Records grain position, area, and equivalent diameter.
7.  Truncated Grain Coordinate Repair: Handles coordinate issues caused by grain truncation at block boundaries.

------------------------------------------------------------------------------------------------------------------------
# Main Parameter Description

Block Processing Parameters

• `block_width`: Block width (default 1436x1436 pixels)

• `overlap`: Overlap area between blocks (default 200 pixels)

• `stride`: Effective advance amount (`block_width` - `overlap`)

Preprocessing Parameters `yuchuli(image, threshold)`

• `threshold`: Binarization threshold (default 150; increasing darkens the image, improving grain boundary clarity but also darkening low-contrast areas)

• `clipLimit`: CLAHE contrast limit (default 2.0)

• `tileGridSize`: CLAHE grid size (default 16x16)

Noise Removal Parameters `remove_noises(binary_img, min_white_area=25, min_black_area=25)`

• `min_white_area`: Minimum area for white noise removal (default 25 pixels)

• `min_black_area`: Minimum area for black noise removal (default 25 pixels)

Region Merging Parameters `merge_small_white_regions(binary_img, region_size, black_threshold, max_white_area, min_area_large, shengzhanghe)`

• `region_size`: Detection window size (default 256 pixels, a sliding detection window)

• `black_threshold`: Black pixel proportion threshold (0-1; if the black pixel proportion in the window exceeds this threshold, a "growth" operation is performed on white patches within this area)

• `max_white_area`: Maximum area for merging white regions (white patches below this value will be grown)

• `min_area_large`: Minimum area to exclude large regions (white patches above this value are not involved in growth, and their boundaries are not affected by the growth of small white patches)

• `shengzhanghe`: Dilation iteration count (higher values result in larger white patch growth)

Grain Identification Parameters `bianyuan(image, left, kernel_min, area_min, block_width=1436, overlap=200)`

• `kernel_min`: Morphological operation kernel size (default 3)

• `area_min`: Minimum grain area (default 25 pixels)

• `block_width`: Block width (default 1436x1436 pixels)

• `overlap`: Overlap area between blocks (default 200 pixels)

------------------------------------------------------------------------------------------------------------------------
# Usage Workflow

# 1. Image Preprocessing

`image = cv2.imread('image.jpg')`

`processed = yuchuli(image, threshold=150)`

# 2. Noise Removal

`clean_image = remove_noises(processed, min_white_area=25, min_black_area=25)`

# 3. Large Image Block Processing

`result = process_large_image('large_image.jpg',`

`                            block_width=1436,`

`                            overlap=200)`

# 4. Region Merging (multiple calls with different parameters)

`merged = merge_small_white_regions(image, 256, 0.4, 10, 10, 1)`

`merged = merge_small_white_regions(merged, 256, 0.3, 20, 20, 5)`

`... more merging steps ...`

# 5. Grain Identification and Data Analysis

`colored_image = bianyuan(processed_image,`

`                        left=50,`

`                        kernel_min=3,`

`                        area_min=25,`

`                        block_width=1436)`

Automatically generates the `grain_data.csv` file.

# 6. Repair Truncated Grains

`fix_fragmented_grains(final_processed_image)`

------------------------------------------------------------------------------------------------------------------------
# Output Files

(1). Preprocessing result: `final_processed.jpg`

(2). Intermediate debug images:

    `large_regions_mask.jpg` (Large region mask)

    `dilated_regions.jpg` (Dilated regions)

    `external_boundary.jpg` (External boundary)

    `small_mask.jpg` (Small region mask)

    `merged_mask.jpg` (Merged mask)

(3). Grain-colored image: `filled_regions.jpg`

(4). Grain data: `grain_data.csv` (Contains ID, X-coordinate, Area, Equivalent Diameter)

------------------------------------------------------------------------------------------------------------------------
# Performance Optimization

(1). Block Processing: Handles ultra-large images (>10000x10000 pixels)

(2). KD-Tree Acceleration: Fast nearest-neighbor search

(3). Sliding Window Optimization: Stride control for processing efficiency

(4). Global Color Mapping: Maintains grain color consistency

(5). Incremental ID Assignment: Prevents grain ID conflicts

------------------------------------------------------------------------------------------------------------------------
# Key Algorithm (Region Merging Algorithm)

(1). Create a large region mask (area > `min_area_large`)

(2). Generate external boundaries (protects large grain boundaries)

(3). Use a sliding window to detect dense black regions

(4). Merge small white regions (area < `max_white_area`)

(5). Apply protected boundaries

------------------------------------------------------------------------------------------------------------------------
# Grain Data Analysis

(1). Equivalent Diameter Calculation: Equivalent Diameter (μm) = 2 × √(Area (pixels) × 2500 / (137×137) / π)

(2). Position Recording: Grain centroid X-coordinate

(3). Area Statistics: Minimum, maximum, and average area

------------------------------------------------------------------------------------------------------------------------
# Usage Recommendations

(1). For new images, start with default parameters.

(2). Adjust the `area_min` parameter based on grain size.

(3). Increase noise removal intensity for poor-quality images.

(4). Adjust block size according to available memory capacity.

(5). Multiple calls to the region merging function can yield better results.

By reasonably adjusting parameters, this system can adapt to the grain analysis needs of various metallographic images, providing accurate grain identification and quantitative data.

P.S. Best results are achieved with high-quality metallographic images.

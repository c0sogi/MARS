import os
import numpy as np
import cv2
import library.config as config
from library.utils import load_or_compute

# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================
# Grid and Voxel Config from library
PC_RANGE = np.array(config.POINT_CLOUD_RANGE, dtype=np.float32)
VOXEL_SIZE = np.array(config.VOXEL_SIZE, dtype=np.float32)
GRID_SIZE = np.array(config.GRID_SIZE, dtype=np.int32)
BEV_CHANNELS = config.BEV_CHANNELS
DOWN_RATIO = config.DOWN_RATIO


# ==============================================================================
# LIDAR LOADING
# ==============================================================================
def load_lidar_points(path):
    """
    Loads LiDAR point cloud from a binary file.
    Assumes data is stored as (N, 4) [x, y, z, intensity] or (N, 5) [x, y, z, i, r].
    Returns (N, 4) array.
    """
    try:
        # Try loading as float32
        points = np.fromfile(path, dtype=np.float32)

        # Heuristic to determine shape
        # Check if divisible by 5 (NuScenes standard)
        if points.shape[0] % 5 == 0:
            points = points.reshape(-1, 5)
            return points[:, :4]
        # Check if divisible by 4 (KITTI standard)
        elif points.shape[0] % 4 == 0:
            points = points.reshape(-1, 4)
            return points
        else:
            # Fallback: assume 4 channels and truncate/pad if necessary (risky but rarely needed)
            # Or raise error
            raise ValueError(
                f"Point cloud file {path} has unexpected size: {points.shape[0]} floats"
            )

    except Exception as e:
        print(f"Error loading LiDAR file {path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)


# ==============================================================================
# BEV RASTERIZATION
# ==============================================================================
def points_to_bev(points):
    """
    Converts a point cloud into a BEV tensor.

    Args:
        points: (N, 4) numpy array [x, y, z, intensity]

    Returns:
        bev_map: (3, H, W) numpy array
                 Channel 0: Max Height (normalized)
                 Channel 1: Mean Intensity (normalized)
                 Channel 2: Log Density (normalized)
    """
    if len(points) == 0:
        return np.zeros((BEV_CHANNELS, GRID_SIZE[1], GRID_SIZE[0]), dtype=np.float32)

    # 1. Filter points outside range
    mask = (
        (points[:, 0] >= PC_RANGE[0])
        & (points[:, 0] < PC_RANGE[3])
        & (points[:, 1] >= PC_RANGE[1])
        & (points[:, 1] < PC_RANGE[4])
        & (points[:, 2] >= PC_RANGE[2])
        & (points[:, 2] < PC_RANGE[5])
    )
    points = points[mask]

    if len(points) == 0:
        return np.zeros((BEV_CHANNELS, GRID_SIZE[1], GRID_SIZE[0]), dtype=np.float32)

    # 2. Compute Grid Indices
    # x_idx = floor((x - x_min) / x_step)
    x_idxs = ((points[:, 0] - PC_RANGE[0]) / VOXEL_SIZE[0]).astype(np.int32)
    y_idxs = ((points[:, 1] - PC_RANGE[1]) / VOXEL_SIZE[1]).astype(np.int32)

    # Clip to ensure safety (though filter should handle it)
    x_idxs = np.clip(x_idxs, 0, GRID_SIZE[0] - 1)
    y_idxs = np.clip(y_idxs, 0, GRID_SIZE[1] - 1)

    # 3. Sort by Flat Index for Grouping
    # flat_idx = y * W + x
    flat_idxs = y_idxs * GRID_SIZE[0] + x_idxs

    # Sort points based on flat index
    sort_order = np.argsort(flat_idxs)
    flat_idxs_sorted = flat_idxs[sort_order]
    points_sorted = points[sort_order]

    # 4. Find unique voxels and their boundaries
    unique_idxs, start_indices = np.unique(flat_idxs_sorted, return_index=True)

    # 5. Compute Stats using reduceat
    # We need end indices for reduceat.
    # np.unique returns the first occurrence.
    # We construct the reduceat indices.

    # Max Height (Z)
    # add.reduceat sums slices. maximum.reduceat computes max of slices.
    max_heights = np.maximum.reduceat(points_sorted[:, 2], start_indices)

    # Mean Intensity
    # sum intensity / count
    sum_intensities = np.add.reduceat(points_sorted[:, 3], start_indices)

    # Counts (Density)
    # We can compute counts by diffing start_indices or using return_counts in unique
    _, counts = np.unique(flat_idxs_sorted, return_counts=True)

    mean_intensities = sum_intensities / counts

    # 6. Fill Map
    bev_map = np.zeros((BEV_CHANNELS, GRID_SIZE[1], GRID_SIZE[0]), dtype=np.float32)

    # Map flat indices back to 2D
    uy_idxs = unique_idxs // GRID_SIZE[0]
    ux_idxs = unique_idxs % GRID_SIZE[0]

    # Channel 0: Max Height
    # Normalize Z to [0, 1] based on range
    z_range = PC_RANGE[5] - PC_RANGE[2]
    norm_heights = (max_heights - PC_RANGE[2]) / z_range
    bev_map[0, uy_idxs, ux_idxs] = norm_heights

    # Channel 1: Mean Intensity
    # Assume intensity is 0-255 or 0-1. If max > 1, normalize by 255.
    if points[:, 3].max() > 1.0:
        norm_intensities = mean_intensities / 255.0
    else:
        norm_intensities = mean_intensities
    bev_map[1, uy_idxs, ux_idxs] = np.clip(norm_intensities, 0, 1)

    # Channel 2: Density
    # Log normalize: log(count + 1) / max_log
    log_counts = np.log1p(counts)
    # Arbitrary scaling factor, assuming max density in a cell rarely exceeds ~100 pts
    norm_density = np.clip(log_counts / 4.0, 0, 1)
    bev_map[2, uy_idxs, ux_idxs] = norm_density

    return bev_map


def get_bev_cached(
    sample_token, lidar_path, cache_dir=config.CACHE_DIR, load_cached=True
):
    """
    Wrapper to load LiDAR, compute BEV, and cache the result.
    Strictly follows the caching logic requirement.
    """

    def compute_fn(path):
        pts = load_lidar_points(path)
        return points_to_bev(pts)

    # Construct cache path
    # We use a subdirectory for BEV to keep things organized
    bev_cache_dir = os.path.join(cache_dir, "bev_maps")
    os.makedirs(bev_cache_dir, exist_ok=True)

    file_path = os.path.join(bev_cache_dir, f"{sample_token}.npy")

    return load_or_compute(
        file_path,
        compute_fn,
        lidar_path,
        load_cached_data=load_cached,
        use_parquet=False,
    )


# ==============================================================================
# TARGET GENERATION (CenterNet Style)
# ==============================================================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute Gaussian radius for a box of size (h, w) such that the IoU
    with the ground truth box is at least min_overlap.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D Gaussian on the heatmap in-place.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian_2d(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_target_maps(boxes_lidar, input_shape=GRID_SIZE, down_ratio=DOWN_RATIO):
    """
    Generates Ground Truth Heatmaps and Regression Maps.

    Args:
        boxes_lidar: (M, 8) numpy array [cx, cy, cz, w, l, h, yaw, class_id]
                     Coordinates must be in LiDAR frame.
        input_shape: (W, H) of the BEV grid (before downsampling).
        down_ratio: Downsampling factor of the backbone.

    Returns:
        heatmap: (NumClasses, H_out, W_out)
        reg_map: (8, H_out, W_out) -> [ox, oy, z, log(w), log(l), log(h), sin, cos]
        reg_mask: (1, H_out, W_out) -> 1 at object centers
    """

    # Output dimensions
    W_out = input_shape[0] // down_ratio
    H_out = input_shape[1] // down_ratio

    num_classes = config.NUM_CLASSES

    # Initialize maps
    heatmap = np.zeros((num_classes, H_out, W_out), dtype=np.float32)
    reg_map = np.zeros((8, H_out, W_out), dtype=np.float32)
    reg_mask = np.zeros((1, H_out, W_out), dtype=np.float32)

    if len(boxes_lidar) == 0:
        return heatmap, reg_map, reg_mask

    # Filter boxes outside range
    # Box center must be within PC_RANGE x/y
    mask = (
        (boxes_lidar[:, 0] >= PC_RANGE[0])
        & (boxes_lidar[:, 0] < PC_RANGE[3])
        & (boxes_lidar[:, 1] >= PC_RANGE[1])
        & (boxes_lidar[:, 1] < PC_RANGE[4])
    )
    boxes_lidar = boxes_lidar[mask]

    for box in boxes_lidar:
        cx, cy, cz, w, l, h, yaw, cls_id = box
        cls_id = int(cls_id)

        # 1. Project center to Output Grid Coordinates
        # Map world (meters) to input grid (pixels)
        x_in = (cx - PC_RANGE[0]) / VOXEL_SIZE[0]
        y_in = (cy - PC_RANGE[1]) / VOXEL_SIZE[1]

        # Map input grid to output grid
        x_out = x_in / down_ratio
        y_out = y_in / down_ratio

        # Integer center
        ct_x = int(x_out)
        ct_y = int(y_out)

        # Check bounds
        if ct_x < 0 or ct_x >= W_out or ct_y < 0 or ct_y >= H_out:
            continue

        # 2. Gaussian Radius
        # Project dimensions to output grid
        h_out_grid = l / VOXEL_SIZE[1] / down_ratio  # Length corresponds to Y in BEV
        w_out_grid = w / VOXEL_SIZE[0] / down_ratio  # Width corresponds to X in BEV

        radius = gaussian_radius((h_out_grid, w_out_grid))
        radius = max(0, int(radius))

        # 3. Draw Heatmap
        draw_gaussian(heatmap[cls_id], (ct_x, ct_y), radius)

        # 4. Fill Regression Targets
        # Local Offset
        reg_map[0, ct_y, ct_x] = x_out - ct_x
        reg_map[1, ct_y, ct_x] = y_out - ct_y

        # Z coordinate
        reg_map[2, ct_y, ct_x] = cz

        # Dimensions (Log space for stability)
        reg_map[3, ct_y, ct_x] = np.log(w + 1e-6)
        reg_map[4, ct_y, ct_x] = np.log(l + 1e-6)
        reg_map[5, ct_y, ct_x] = np.log(h + 1e-6)

        # Orientation
        reg_map[6, ct_y, ct_x] = np.sin(yaw)
        reg_map[7, ct_y, ct_x] = np.cos(yaw)

        # Mask
        reg_mask[0, ct_y, ct_x] = 1.0

    return heatmap, reg_map, reg_mask

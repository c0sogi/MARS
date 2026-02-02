import numpy as np
import torch
import math
from library.config import VoxelConfig


def gaussian_radius(height, width, min_overlap=0.7):
    """
    Calculate the radius for the Gaussian kernel based on object dimensions and overlap.
    """
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


def gaussian_2d(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D Gaussian on the heatmap at the specified center.
    Args:
        heatmap: (H, W) array to update in-place
        center: (x, y) coordinates
        radius: Gaussian radius
        k: Amplitude (default 1)
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


def create_voxel_grid(points, config=None):
    """
    Voxelizes a point cloud into pillars.
    Args:
        points: (N, 4) [x, y, z, intensity]
        config: VoxelConfig class or instance
    Returns:
        pillar_features: (M, max_points, 9) [x, y, z, i, x_c, y_c, z_c, x_p, y_p]
        pillar_coords: (M, 3) [z_idx, y_idx, x_idx]
        pillar_num_points: (M,) Number of valid points in each pillar
    """
    if config is None:
        config = VoxelConfig

    # 1. Filter points outside range
    pc_range = np.array(config.point_cloud_range)
    voxel_size = np.array(config.voxel_size)
    grid_size = np.array(config.grid_size)

    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    if points.shape[0] == 0:
        return (
            np.zeros(
                (0, config.max_points_per_pillar, config.num_point_features),
                dtype=np.float32,
            ),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    # 2. Calculate coordinates
    coords = ((points[:, :3] - pc_range[:3]) / voxel_size).astype(np.int32)

    # 3. Sort by pillar ID to group points
    # Key: z * (H*W) + y * W + x
    # Note: For pillars, z is usually 0, but we keep it generic
    key = (
        coords[:, 2] * (grid_size[1] * grid_size[0])
        + coords[:, 1] * grid_size[0]
        + coords[:, 0]
    )

    sort_idx = np.argsort(key)
    points = points[sort_idx]
    coords = coords[sort_idx]
    key = key[sort_idx]

    # 4. Identify unique pillars
    _, unique_indices, counts = np.unique(key, return_index=True, return_counts=True)

    # 5. Limit number of pillars
    # Use max_pillars_train as default limit logic
    max_pillars = config.max_pillars_train
    num_pillars = min(len(unique_indices), max_pillars)

    unique_indices = unique_indices[:num_pillars]
    counts = counts[:num_pillars]

    # 6. Filter points to only those in selected pillars
    total_points_in_pillars = unique_indices[-1] + counts[-1] if num_pillars > 0 else 0
    points = points[:total_points_in_pillars]
    # coords = coords[:total_points_in_pillars] # Not strictly needed for feature calc

    # 7. Prepare Output Tensors
    pillar_features = np.zeros(
        (num_pillars, config.max_points_per_pillar, config.num_point_features),
        dtype=np.float32,
    )
    pillar_coords = np.zeros((num_pillars, 3), dtype=np.int32)
    pillar_num_points = np.zeros((num_pillars,), dtype=np.int32)

    # Fill pillar coords (z, y, x)
    # Take the coordinate of the first point in each pillar
    representative_coords = coords[unique_indices]
    pillar_coords[:] = representative_coords[:, [2, 1, 0]]

    # 8. Vectorized Feature Filling
    # Map each point to its pillar index (0..num_pillars-1)
    pillar_ids = np.repeat(np.arange(num_pillars), counts)

    # Calculate offset of each point within its pillar
    point_indices = np.arange(total_points_in_pillars)
    offsets = point_indices - unique_indices[pillar_ids]

    # Mask points that exceed max_points_per_pillar
    mask_p = offsets < config.max_points_per_pillar

    valid_points = points[mask_p]
    valid_pillar_ids = pillar_ids[mask_p]
    valid_offsets = offsets[mask_p]

    # Fill raw features (x, y, z, i)
    dim_p = valid_points.shape[1]
    pillar_features[valid_pillar_ids, valid_offsets, :dim_p] = valid_points

    # Update num_points per pillar
    pillar_num_points[:] = np.minimum(counts, config.max_points_per_pillar)

    # 9. Calculate Derived Features (Means and Centers)
    # Sum x, y, z for each pillar to compute arithmetic mean
    sum_xyz = np.zeros((num_pillars, 3), dtype=np.float32)
    np.add.at(sum_xyz, valid_pillar_ids, valid_points[:, :3])

    means = sum_xyz / pillar_num_points[:, None]  # (M, 3)

    # Calculate geometric centers of the voxels
    vx, vy, vz = voxel_size
    x_min, y_min, z_min = pc_range[:3]

    # pillar_coords is (z, y, x)
    cx = pillar_coords[:, 2] * vx + x_min + vx / 2
    cy = pillar_coords[:, 1] * vy + y_min + vy / 2
    cz = pillar_coords[:, 0] * vz + z_min + vz / 2

    centers = np.stack([cx, cy, cz], axis=1)  # (M, 3)

    # Broadcast means and centers to points
    p_means = means[valid_pillar_ids]
    p_centers = centers[valid_pillar_ids]

    # Feature: x - xc, y - yc, z - zc (Offset from arithmetic mean)
    pillar_features[valid_pillar_ids, valid_offsets, 4:7] = (
        valid_points[:, :3] - p_means
    )

    # Feature: x - xp, y - yp (Offset from geometric center)
    # Note: usually only x and y offsets are used for pillars, but we fill 2 dims
    pillar_features[valid_pillar_ids, valid_offsets, 7:9] = (
        valid_points[:, :2] - p_centers[:, :2]
    )

    return pillar_features, pillar_coords, pillar_num_points


class BoxUtils:
    @staticmethod
    def quaternion_to_matrix(quaternions):
        """
        Convert quaternions to rotation matrices.
        Args:
            quaternions: (N, 4) or (4,) [w, x, y, z]
        Returns:
            matrices: (N, 3, 3) or (3, 3)
        """
        quaternions = np.asarray(quaternions)
        if quaternions.ndim == 1:
            quaternions = quaternions[None, :]

        w, x, y, z = (
            quaternions[:, 0],
            quaternions[:, 1],
            quaternions[:, 2],
            quaternions[:, 3],
        )

        N = quaternions.shape[0]
        matrices = np.zeros((N, 3, 3), dtype=quaternions.dtype)

        matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        matrices[:, 0, 1] = 2 * (x * y - z * w)
        matrices[:, 0, 2] = 2 * (x * z + y * w)

        matrices[:, 1, 0] = 2 * (x * y + z * w)
        matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        matrices[:, 1, 2] = 2 * (y * z - x * w)

        matrices[:, 2, 0] = 2 * (x * z - y * w)
        matrices[:, 2, 1] = 2 * (y * z + x * w)
        matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        if N == 1:
            return matrices[0]
        return matrices

    @staticmethod
    def get_corners(boxes):
        """
        Get 8 corners of bounding boxes.
        Args:
            boxes: (N, 7) [x, y, z, w, l, h, yaw]
        Returns:
            corners: (N, 8, 3)
        """
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            boxes = boxes[None, :]

        # Unpack
        # Assuming format: x, y, z, w, l, h, yaw
        # w: width (x-axis dim), l: length (y-axis dim), h: height (z-axis dim)
        x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        w, l, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
        yaw = boxes[:, 6]

        x_corners = l / 2
        y_corners = w / 2
        z_corners = h / 2

        # (N, 8, 3)
        corners = np.zeros((boxes.shape[0], 8, 3), dtype=boxes.dtype)

        # Standard corner template relative to center
        # Top face
        corners[:, 0] = np.stack([x_corners, y_corners, z_corners], axis=1)
        corners[:, 1] = np.stack([x_corners, -y_corners, z_corners], axis=1)
        corners[:, 2] = np.stack([-x_corners, -y_corners, z_corners], axis=1)
        corners[:, 3] = np.stack([-x_corners, y_corners, z_corners], axis=1)
        # Bottom face
        corners[:, 4] = np.stack([x_corners, y_corners, -z_corners], axis=1)
        corners[:, 5] = np.stack([x_corners, -y_corners, -z_corners], axis=1)
        corners[:, 6] = np.stack([-x_corners, -y_corners, -z_corners], axis=1)
        corners[:, 7] = np.stack([-x_corners, y_corners, -z_corners], axis=1)

        # Rotate
        c = np.cos(yaw)
        s = np.sin(yaw)

        # Apply rotation around Z
        x_new = corners[..., 0] * c[:, None] - corners[..., 1] * s[:, None]
        y_new = corners[..., 0] * s[:, None] + corners[..., 1] * c[:, None]

        corners[..., 0] = x_new
        corners[..., 1] = y_new

        # Translate
        corners += boxes[:, None, :3]

        return corners

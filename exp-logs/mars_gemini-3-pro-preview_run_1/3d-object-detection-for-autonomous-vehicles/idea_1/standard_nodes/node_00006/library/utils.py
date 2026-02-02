import os
import sys
import logging
import numpy as np
import pandas as pd
import cv2
import library.config as config


def get_logger(name=__name__):
    """
    Creates and configures a logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_or_compute(
    cache_path, compute_fn, *args, load_cached_data=True, use_parquet=False, **kwargs
):
    """
    Generic caching utility.

    Args:
        cache_path (str): Path to save/load the cached file.
        compute_fn (callable): Function to compute data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        use_parquet (bool): If True, uses pd.to_parquet/read_parquet. Else uses np.save/load.
        *args, **kwargs: Arguments passed to compute_fn.

    Returns:
        The loaded or computed data.
    """
    # Ensure directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            if use_parquet:
                data = pd.read_parquet(cache_path)
            else:
                data = np.load(cache_path, allow_pickle=True)
                # Unwrap 0-d array if necessary (common with np.save of object)
                if data.shape == ():
                    data = data.item()
            return data
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # Compute
    data = compute_fn(*args, **kwargs)

    # Save
    try:
        if use_parquet:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(cache_path, index=False)
            else:
                raise ValueError("Data must be a DataFrame to save as parquet.")
        else:
            np.save(cache_path, data)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return data


# ==============================================================================
# GEOMETRIC TRANSFORMATIONS
# ==============================================================================


def quaternion_to_matrix(q):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def get_transform_matrix(translation, rotation_quat):
    """
    Construct a 4x4 transformation matrix from translation and quaternion.

    Args:
        translation: list or array [x, y, z]
        rotation_quat: list or array [w, x, y, z]

    Returns:
        4x4 numpy array
    """
    R = quaternion_to_matrix(rotation_quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def transform_points(points, matrix):
    """
    Apply a 4x4 rigid transformation matrix to 3D points.

    Args:
        points: (N, 3) numpy array
        matrix: (4, 4) numpy array

    Returns:
        (N, 3) numpy array of transformed points
    """
    # Add homogeneous coordinate
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))

    # Transform: (4, 4) @ (4, N) -> (4, N) -> Transpose back to (N, 4)
    transformed_hom = (matrix @ points_hom.T).T

    # Return x, y, z
    return transformed_hom[:, :3]


# ==============================================================================
# METRICS & IOU
# ==============================================================================


def get_corners_2d(box):
    """
    Get the 4 corners of a 2D rotated box.

    Args:
        box: [center_x, center_y, width, length, yaw]
             Note: width is size along x-axis (before rotation), length along y-axis.

    Returns:
        (4, 2) numpy array of corners.
    """
    cx, cy, w, l, yaw = box

    # Corners relative to center
    # Counter-clockwise from bottom-left (assuming standard cartesian)
    # x: width, y: length
    x_corners = np.array([-w / 2, w / 2, w / 2, -w / 2])
    y_corners = np.array([-l / 2, -l / 2, l / 2, l / 2])

    # Rotate
    c = np.cos(yaw)
    s = np.sin(yaw)

    x_rot = c * x_corners - s * y_corners
    y_rot = s * x_corners + c * y_corners

    # Translate
    x_final = x_rot + cx
    y_final = y_rot + cy

    corners = np.vstack((x_final, y_final)).T
    return corners.astype(np.float32)


def box_3d_iou(box_a, box_b):
    """
    Calculate 3D Intersection over Union between two boxes.

    Args:
        box_a, box_b: [cx, cy, cz, w, l, h, yaw]

    Returns:
        float: IoU value
    """
    # 1. Height Intersection
    # box format: z is center
    za_min = box_a[2] - box_a[5] / 2
    za_max = box_a[2] + box_a[5] / 2
    zb_min = box_b[2] - box_b[5] / 2
    zb_max = box_b[2] + box_b[5] / 2

    inter_h_min = max(za_min, zb_min)
    inter_h_max = min(za_max, zb_max)
    inter_h = max(0.0, inter_h_max - inter_h_min)

    if inter_h == 0:
        return 0.0

    # 2. BEV Intersection (Area)
    # Extract 2D parameters: cx, cy, w, l, yaw
    # Indices: 0, 1, 3, 4, 6
    rect_a = get_corners_2d([box_a[0], box_a[1], box_a[3], box_a[4], box_a[6]])
    rect_b = get_corners_2d([box_b[0], box_b[1], box_b[3], box_b[4], box_b[6]])

    # Use OpenCV for polygon intersection
    # intersectConvexConvex returns (area, intersection_polygon)
    # We only need area.
    try:
        area_inter, _ = cv2.intersectConvexConvex(rect_a, rect_b)
    except Exception:
        # Fallback if geometry is degenerate
        area_inter = 0.0

    if area_inter == 0:
        return 0.0

    # 3. Volume Calculation
    vol_inter = area_inter * inter_h

    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]

    vol_union = vol_a + vol_b - vol_inter

    return vol_inter / vol_union if vol_union > 0 else 0.0

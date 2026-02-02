import os
import json
import numpy as np
import pandas as pd
from library.config import Config


def load_table(data_dir, table_name, load_cached_data=True):
    """
    Loads a NuScenes-style JSON table, with caching to Parquet.

    Args:
        data_dir (str): Directory containing the JSON file (e.g., train_data).
        table_name (str): Name of the table (e.g., 'sample_data').
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: The loaded table.
    """
    # Determine split name based on directory for cache naming to prevent collisions
    if "train" in data_dir:
        split = "train"
    elif "test" in data_dir:
        split = "test"
    else:
        split = "other"

    cache_filename = f"{split}_{table_name}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            # If cache is corrupt, fallback to source
            pass

    # 2. Load from JSON
    json_path = os.path.join(data_dir, f"{table_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Table {table_name} not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 3. Save to Cache
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        # If caching fails, just proceed with the dataframe
        pass

    return df


def get_quaternion_from_yaw(yaw):
    """
    Convert a yaw angle (rotation around Z) to a quaternion [w, x, y, z].
    """
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def get_yaw_from_quaternion(q):
    """
    Extract yaw (rotation around Z-axis) from a quaternion [w, x, y, z].
    """
    w, x, y, z = q
    # Standard conversion from quaternion to Euler yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def get_transformation_matrix(translation, rotation):
    """
    Construct a 4x4 transformation matrix from translation and rotation.

    Args:
        translation (list/array): [x, y, z]
        rotation (list/array): Quaternion [w, x, y, z]

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    t = np.array(translation)
    q = np.array(rotation)
    w, x, y, z = q

    # Rotation matrix from quaternion
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )

    # Construct 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def transform_points(points, matrix):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.
    Preserves additional feature channels (intensity, etc.).

    Args:
        points (np.ndarray): Shape (N, C) where C >= 3.
        matrix (np.ndarray): Shape (4, 4).

    Returns:
        np.ndarray: Transformed points, shape (N, C).
    """
    if points.shape[0] == 0:
        return points

    # Extract XYZ
    xyz = points[:, :3]

    # Homogeneous coordinates: (N, 4)
    ones = np.ones((xyz.shape[0], 1))
    xyz_hom = np.hstack((xyz, ones))

    # Transform: xyz_hom @ Matrix.T
    xyz_trans = xyz_hom @ matrix.T

    # Update XYZ in original array copy
    points_trans = points.copy()
    points_trans[:, :3] = xyz_trans[:, :3]

    return points_trans


def transform_box_to_global(box, ego_translation, ego_rotation):
    """
    Transforms a bounding box from ego frame to global frame.

    Args:
        box (list/array): [center_x, center_y, center_z, w, l, h, yaw]
        ego_translation (list/array): [x, y, z] of ego vehicle.
        ego_rotation (list/array): [w, x, y, z] quaternion of ego vehicle.

    Returns:
        np.ndarray: Transformed box parameters [center_x, center_y, center_z, w, l, h, yaw]
    """
    # 1. Transform Center
    center = np.array([box[0], box[1], box[2]]).reshape(1, 3)
    ego_mat = get_transformation_matrix(ego_translation, ego_rotation)

    # Use transform_points but extract only the coordinate part
    global_center_full = transform_points(center, ego_mat)
    global_center = global_center_full[0, :3]

    # 2. Transform Yaw
    # Global Yaw = Local Yaw + Ego Yaw
    ego_yaw = get_yaw_from_quaternion(ego_rotation)
    global_yaw = box[6] + ego_yaw

    # Normalize yaw to [-pi, pi]
    global_yaw = (global_yaw + np.pi) % (2 * np.pi) - np.pi

    return np.array(
        [
            global_center[0],
            global_center[1],
            global_center[2],
            box[3],
            box[4],
            box[5],
            global_yaw,
        ]
    )


def load_point_cloud(file_path):
    """
    Robustly load point cloud from binary file, handling variable dimensions.

    Args:
        file_path (str): Path to .bin file.

    Returns:
        np.ndarray: Point cloud array (N, D).
    """
    if not os.path.exists(file_path):
        return np.zeros((0, 5), dtype=np.float32)

    points = np.fromfile(file_path, dtype=np.float32)

    # Heuristic reshape based on total elements
    if points.shape[0] % 5 == 0:
        return points.reshape(-1, 5)
    elif points.shape[0] % 4 == 0:
        return points.reshape(-1, 4)
    elif points.shape[0] % 3 == 0:
        return points.reshape(-1, 3)
    else:
        # Fallback: try 5, truncate if necessary
        dim = 5
        num_points = points.shape[0] // dim
        return points[: num_points * dim].reshape(num_points, dim)

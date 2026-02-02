import os
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple


def load_json_table(json_path: str) -> pd.DataFrame:
    """
    Loads a JSON file containing a list of records into a DataFrame.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def quaternion_to_euler(q: List[float]) -> Tuple[float, float, float]:
    """
    Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """
    Convert Euler angles to quaternion [w, x, y, z].
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def get_transformation_matrix(
    translation: List[float], rotation: List[float]
) -> np.ndarray:
    """
    Constructs a 4x4 homogeneous transformation matrix from translation [x, y, z]
    and quaternion rotation [w, x, y, z].
    """
    t = np.array(translation)
    q = np.array(rotation)  # [w, x, y, z]

    w, x, y, z = q

    # Rotation matrix from quaternion
    # R = [ 1-2y^2-2z^2   2xy-2wz       2xz+2wy
    #       2xy+2wz       1-2x^2-2z^2   2yz-2wx
    #       2xz-2wy       2yz+2wx       1-2x^2-2y^2 ]

    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )

    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = t
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to a set of 3D points (N, 3).
    """
    if points.shape[0] == 0:
        return points

    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))

    # Apply transformation: (Matrix @ Points.T).T
    transformed_hom = (matrix @ points_hom.T).T

    # Return to cartesian (N, 3)
    return transformed_hom[:, :3]


def world_to_sensor(
    points: np.ndarray, ego_pose: Dict, calibrated_sensor: Dict
) -> np.ndarray:
    """
    Transforms points from Global World frame to Sensor frame.
    Chain: World -> Ego -> Sensor
    """
    # Matrix: Ego -> World
    m_ego_to_global = get_transformation_matrix(
        ego_pose["translation"], ego_pose["rotation"]
    )

    # Matrix: Sensor -> Ego
    m_sensor_to_ego = get_transformation_matrix(
        calibrated_sensor["translation"], calibrated_sensor["rotation"]
    )

    # Inverse to go World -> Ego
    m_global_to_ego = np.linalg.inv(m_ego_to_global)

    # Inverse to go Ego -> Sensor
    m_ego_to_sensor = np.linalg.inv(m_sensor_to_ego)

    # Combined: World -> Sensor
    # P_sensor = M_ego_to_sensor * M_global_to_ego * P_world
    m_global_to_sensor = m_ego_to_sensor @ m_global_to_ego

    return transform_points(points, m_global_to_sensor)


def sensor_to_world(
    points: np.ndarray, ego_pose: Dict, calibrated_sensor: Dict
) -> np.ndarray:
    """
    Transforms points from Sensor frame to Global World frame.
    Chain: Sensor -> Ego -> World
    """
    # Matrix: Ego -> World
    m_ego_to_global = get_transformation_matrix(
        ego_pose["translation"], ego_pose["rotation"]
    )

    # Matrix: Sensor -> Ego
    m_sensor_to_ego = get_transformation_matrix(
        calibrated_sensor["translation"], calibrated_sensor["rotation"]
    )

    # Combined: Sensor -> World
    # P_world = M_ego_to_global * M_sensor_to_ego * P_sensor
    m_sensor_to_global = m_ego_to_global @ m_sensor_to_ego

    return transform_points(points, m_sensor_to_global)


def compute_iou_bev(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes the Intersection over Union (IoU) between two sets of axis-aligned 2D boxes.

    Args:
        boxes_a: (N, 4) array of [center_x, center_y, width, length]
        boxes_b: (M, 4) array of [center_x, center_y, width, length]

    Returns:
        (N, M) array of IoU values.
    """
    # Convert [cx, cy, w, l] to [x_min, y_min, x_max, y_max]
    # N x 4
    box_a_min = boxes_a[:, :2] - boxes_a[:, 2:] / 2
    box_a_max = boxes_a[:, :2] + boxes_a[:, 2:] / 2
    area_a = boxes_a[:, 2] * boxes_a[:, 3]

    # M x 4
    box_b_min = boxes_b[:, :2] - boxes_b[:, 2:] / 2
    box_b_max = boxes_b[:, :2] + boxes_b[:, 2:] / 2
    area_b = boxes_b[:, 2] * boxes_b[:, 3]

    # Expand dims for broadcasting: (N, 1, 2) vs (1, M, 2)
    inter_min = np.maximum(box_a_min[:, None, :], box_b_min[None, :, :])
    inter_max = np.minimum(box_a_max[:, None, :], box_b_max[None, :, :])

    # Compute Intersection Dimensions
    inter_dims = np.maximum(inter_max - inter_min, 0)

    # Compute Intersection Area
    inter_area = inter_dims[:, :, 0] * inter_dims[:, :, 1]

    # Compute Union Area
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    # Avoid division by zero
    union_area = np.maximum(union_area, 1e-6)

    return inter_area / union_area

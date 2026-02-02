import numpy as np
import math


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


def get_pose_matrix(translation, rotation, inverse=False):
    """
    Construct a 4x4 homogeneous transformation matrix.
    Args:
        translation: (x, y, z)
        rotation: quaternion (w, x, y, z)
        inverse: If True, return the inverse transformation (Global/Ego -> Local).
    """
    R = quaternion_to_matrix(rotation)
    T = np.array(translation)

    mat = np.eye(4)
    if inverse:
        # Inverse of [R|t] is [R^T | -R^T * t]
        mat[:3, :3] = R.T
        mat[:3, 3] = -R.T @ T
    else:
        mat[:3, :3] = R
        mat[:3, 3] = T

    return mat


def project_3d_to_2d(points_3d, intrinsic):
    """
    Project 3D points in camera coordinates to 2D image coordinates.
    Args:
        points_3d: (N, 3) or (3,) array of points (x, y, z)
        intrinsic: (3, 3) camera intrinsic matrix
    Returns:
        points_2d: (N, 2) projected points (u, v)
        depths: (N,) depths (z)
    """
    points_3d = np.atleast_2d(points_3d)

    # Perspective projection: x_img = K * X_cam
    # points_3d is (N, 3), we need (3, N) for matrix multiplication
    points_img = (intrinsic @ points_3d.T).T  # (N, 3)

    # Normalize by Z
    depths = points_img[:, 2]

    # Avoid division by zero
    safe_depths = np.maximum(depths, 1e-5)

    u = points_img[:, 0] / safe_depths
    v = points_img[:, 1] / safe_depths

    points_2d = np.stack([u, v], axis=1)

    return points_2d, depths


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculate the radius for the Gaussian kernel based on object size and IoU overlap.
    Standard CenterNet implementation.
    Args:
        det_size: (height, width) of the bounding box in pixels
        min_overlap: minimum IoU overlap
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


def gaussian_2d(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw a Gaussian splat on the heatmap.
    Args:
        heatmap: (H, W) array
        center: (x, y) integer coordinates
        radius: float radius
        k: scaling factor (usually 1 for ground truth)
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


def global_to_camera(
    points_global, ego_translation, ego_rotation, sensor_translation, sensor_rotation
):
    """
    Transform points from Global frame to Camera frame.

    Chain: Global -> Ego -> Sensor (Camera)

    Args:
        points_global: (N, 3)
        ego_translation: (x, y, z)
        ego_rotation: (w, x, y, z)
        sensor_translation: (x, y, z) relative to ego
        sensor_rotation: (w, x, y, z) relative to ego
    """
    # 1. Global to Ego
    # P_ego = R_ego_inv * (P_global - T_ego)
    # Using row vectors: P_ego = (P_global - T_ego) @ R_ego
    # (since R_ego is rotation from Ego to Global, R_ego^T is inverse)
    R_ego = quaternion_to_matrix(ego_rotation)
    points_ego = (points_global - np.array(ego_translation)) @ R_ego

    # 2. Ego to Sensor
    # P_sensor = R_sensor_inv * (P_ego - T_sensor)
    R_sensor = quaternion_to_matrix(sensor_rotation)
    points_camera = (points_ego - np.array(sensor_translation)) @ R_sensor

    return points_camera


def camera_to_global(
    points_camera, ego_translation, ego_rotation, sensor_translation, sensor_rotation
):
    """
    Transform points from Camera frame to Global frame.

    Chain: Camera -> Ego -> Global

    Args:
        points_camera: (N, 3)
        ego_translation: (x, y, z)
        ego_rotation: (w, x, y, z)
        sensor_translation: (x, y, z) relative to ego
        sensor_rotation: (w, x, y, z) relative to ego
    """
    # 1. Sensor to Ego
    # P_ego = R_sensor * P_camera + T_sensor
    # Row vectors: P_ego = P_camera @ R_sensor.T + T_sensor
    R_sensor = quaternion_to_matrix(sensor_rotation)
    points_ego = points_camera @ R_sensor.T + np.array(sensor_translation)

    # 2. Ego to Global
    # P_global = R_ego * P_ego + T_ego
    R_ego = quaternion_to_matrix(ego_rotation)
    points_global = points_ego @ R_ego.T + np.array(ego_translation)

    return points_global

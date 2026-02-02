import os
import random
import numpy as np
import torch
import logging
import cv2
import math
import json
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(log_file=None):
    """
    Creates and returns a logger that logs to both console and a file (if specified).
    """
    logger = logging.getLogger("dla_3d_det")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def quaternion_yaw(q):
    """
    Calculate yaw angle from a quaternion [w, x, y, z].
    """
    w, x, y, z = q

    # Yaw is rotation around Z-axis
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


def get_corners(x, y, z, w, l, h, yaw):
    """
    Get 8 corners of a 3D bounding box.
    Args:
        x, y, z: Center coordinates
        w, l, h: Dimensions (width, length, height)
        yaw: Angle in radians around Z-axis
    Returns:
        corners: (8, 3) numpy array
    """
    # Half dimensions
    dx = w / 2.0
    dy = l / 2.0
    dz = h / 2.0

    # Base corners centered at origin
    x_corners = np.array([dx, -dx, -dx, dx, dx, -dx, -dx, dx])
    y_corners = np.array([dy, dy, -dy, -dy, dy, dy, -dy, -dy])
    z_corners = np.array([dz, dz, dz, dz, -dz, -dz, -dz, -dz])

    # Rotation matrix around Z-axis
    c = np.cos(yaw)
    s = np.sin(yaw)

    # Rotate
    x_rot = x_corners * c - y_corners * s
    y_rot = x_corners * s + y_corners * c
    z_rot = z_corners

    # Translate
    corners = np.vstack((x_rot + x, y_rot + y, z_rot + z)).T
    return corners


def box_to_corners(box):
    """
    Wrapper for get_corners taking a list/array.
    box: [x, y, z, w, l, h, yaw]
    """
    return get_corners(box[0], box[1], box[2], box[3], box[4], box[5], box[6])


def calc_iou_3d(box1, box2):
    """
    Calculate 3D IoU as defined in the task:
    IoU = (Area_Inter * Height_Inter) / (Vol_1 + Vol_2 - Area_Inter * Height_Inter)

    Args:
        box1, box2: [x, y, z, w, l, h, yaw]
    """
    # 1. Calculate BEV Intersection Area
    # Create rotated rectangles for OpenCV: ((cx, cy), (w, l), angle_deg)
    # Note: OpenCV angle is degrees.
    rect1 = (
        (float(box1[0]), float(box1[1])),
        (float(box1[3]), float(box1[4])),
        np.degrees(float(box1[6])),
    )
    rect2 = (
        (float(box2[0]), float(box2[1])),
        (float(box2[3]), float(box2[4])),
        np.degrees(float(box2[6])),
    )

    inter_area = 0.0
    try:
        ret, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
        if ret == cv2.INTERSECT_FULL:
            # One rectangle is fully inside the other
            area1 = box1[3] * box1[4]
            area2 = box2[3] * box2[4]
            inter_area = min(area1, area2)
        elif ret != cv2.INTERSECT_NONE and inter_pts is not None:
            # Partial intersection
            hull = cv2.convexHull(inter_pts)
            inter_area = cv2.contourArea(hull)
    except Exception:
        # Fallback if geometry fails
        inter_area = 0.0

    # 2. Calculate Height Intersection
    z1_min = box1[2] - box1[5] / 2.0
    z1_max = box1[2] + box1[5] / 2.0
    z2_min = box2[2] - box2[5] / 2.0
    z2_max = box2[2] + box2[5] / 2.0

    z_inter_min = max(z1_min, z2_min)
    z_inter_max = min(z1_max, z2_max)

    h_inter = max(0.0, z_inter_max - z_inter_min)

    # 3. Calculate 3D IoU
    vol_inter = inter_area * h_inter

    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]

    union_vol = vol1 + vol2 - vol_inter

    if union_vol <= 1e-6:
        return 0.0

    return vol_inter / union_vol


def save_cache(data, path):
    """
    Saves data to disk. Uses .npz for numpy arrays/dicts of arrays,
    and .json for other serializable data.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Determine type and save
    if isinstance(data, (np.ndarray, dict)):
        # Check if dict contains only arrays (suitable for npz)
        if isinstance(data, dict):
            np.savez(path, **data)
        else:
            np.save(path, data)
    else:
        # Fallback to JSON for metadata lists/dicts
        # Ensure path ends with .json if not present
        if not path.endswith(".json"):
            path += ".json"
        with open(path, "w") as f:
            json.dump(data, f)


def load_cache(path):
    """
    Loads data from disk. Checks for .npz, .npy, and .json extensions.
    """
    # Try exact path
    if os.path.exists(path):
        if path.endswith(".npy") or path.endswith(".npz"):
            return np.load(path, allow_pickle=False)  # Strict no pickle
        elif path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)

    # Try appending extensions
    if os.path.exists(path + ".npz"):
        return np.load(path + ".npz")
    elif os.path.exists(path + ".npy"):
        return np.load(path + ".npy")
    elif os.path.exists(path + ".json"):
        with open(path + ".json", "r") as f:
            return json.load(f)

    return None

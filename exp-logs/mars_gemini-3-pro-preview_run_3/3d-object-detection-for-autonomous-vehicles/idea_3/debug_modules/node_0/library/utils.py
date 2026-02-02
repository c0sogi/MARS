import numpy as np
import torch
import cv2
import logging
import sys
import os
import math
from library.config import ANCHOR_CONFIGS


def setup_logger(log_file=None):
    """
    Sets up a logger that outputs to console and optionally a file.
    """
    logger = logging.getLogger("PointPillars")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_corners_2d(box):
    """
    Calculates the 4 corners of a 2D rotated box.
    Args:
        box: [x, y, w, l, yaw] or [x, y, z, w, l, h, yaw]
    Returns:
        corners: (4, 2) numpy array of (x, y) coordinates
    """
    # Extract parameters
    if len(box) == 7:
        x, y, _, w, l, _, yaw = box
    else:
        x, y, w, l, yaw = box

    # Rotation matrix
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    # Local corners: assuming Length is along the heading (Y-axis in local frame if Yaw=0 points Y)
    # or Length is along X if Yaw=0 points X.
    # Based on standard conventions:
    # Yaw is rotation around Z.
    # We assume the box is centered at 0,0.
    # Dimensions are w (width) and l (length).
    # Corners: (+w/2, +l/2), (+w/2, -l/2), (-w/2, -l/2), (-w/2, +l/2)
    # Note: Whether L is x or y depends on definition.
    # Usually L is along the object's forward axis.

    # Let's assume standard geometric center-to-corner:
    # x_corners = [l/2, l/2, -l/2, -l/2]
    # y_corners = [w/2, -w/2, -w/2, w/2]
    # This assumes L is along X-axis before rotation.
    # If Yaw=0 points East (X), then L is along X.

    # Using half dimensions
    h_w = w / 2
    h_l = l / 2

    # Define corners in local frame (assuming Length is along X-axis for 0 yaw)
    # If the car points along X, length is X-dim.
    corners_local = np.array([[h_l, h_w], [h_l, -h_w], [-h_l, -h_w], [-h_l, h_w]])

    # Rotate and translate
    corners_global = corners_local @ R.T + np.array([x, y])
    return corners_global.astype(np.float32)


def box_iou_3d_pair(box_a, box_b):
    """
    Calculates 3D IoU for a single pair of boxes.
    Args:
        box_a, box_b: [x, y, z, w, l, h, yaw]
    Returns:
        iou: float
    """
    # 1. Height Overlap
    # z is center. z_min = z - h/2, z_max = z + h/2
    za_min = box_a[2] - box_a[5] / 2
    za_max = box_a[2] + box_a[5] / 2
    zb_min = box_b[2] - box_b[5] / 2
    zb_max = box_b[2] + box_b[5] / 2

    inter_h_min = max(za_min, zb_min)
    inter_h_max = min(za_max, zb_max)
    inter_h = max(0.0, inter_h_max - inter_h_min)

    if inter_h == 0:
        return 0.0

    # 2. BEV Intersection (2D)
    rect_a = get_corners_2d(box_a)
    rect_b = get_corners_2d(box_b)

    # Use OpenCV to find intersection area
    # intersectConvexConvex returns (area, intersection_points)
    # But the python wrapper returns (area, points) or just area depending on version/usage
    # Actually cv2.intersectConvexConvex returns (float area, ndarray points)
    try:
        area_inter, _ = cv2.intersectConvexConvex(rect_a, rect_b)
    except Exception:
        # Fallback for degenerate geometry
        area_inter = 0.0

    if area_inter <= 0:
        return 0.0

    # 3. 3D IoU
    vol_inter = area_inter * inter_h

    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]

    vol_union = vol_a + vol_b - vol_inter

    return vol_inter / (vol_union + 1e-6)


def nms_3d(boxes, scores, iou_threshold=0.1, max_dets=500):
    """
    Performs Non-Maximum Suppression on 3D boxes.
    Args:
        boxes: (N, 7) torch.Tensor or numpy array [x, y, z, w, l, h, yaw]
        scores: (N,) torch.Tensor or numpy array
        iou_threshold: float
        max_dets: int
    Returns:
        keep_indices: List of indices to keep
    """
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = scores.argsort()[::-1]
    keep = []

    # Limit candidates for speed if necessary (optional, but good for very large N)
    # order = order[:2000]

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_dets:
            break

        if order.size == 1:
            break

        # Compare current box (i) with rest
        rest_indices = order[1:]
        rest_boxes = boxes[rest_indices]
        current_box = boxes[i]

        # We need to compute IoU between current_box and all rest_boxes
        # Since python loop is slow, we can try to vectorize or just loop
        # Given N is usually small after score filtering (<1000), a loop is okay-ish
        # but pure python loop for 1 vs N is better than N vs N.

        # Optimization: Filter by simple BEV bounding box overlap first
        # Current BEV bounds
        cx, cy, _, w, l, _, _ = current_box
        radius = math.sqrt(w**2 + l**2) / 2

        # Rest BEV bounds approx
        r_cx = rest_boxes[:, 0]
        r_cy = rest_boxes[:, 1]
        r_w = rest_boxes[:, 3]
        r_l = rest_boxes[:, 4]
        r_radius = np.sqrt(r_w**2 + r_l**2) / 2

        dist = np.sqrt((cx - r_cx) ** 2 + (cy - r_cy) ** 2)

        # Only check IoU if distance is small enough
        potential_overlap_mask = dist < (radius + r_radius)

        keep_mask = np.ones(len(rest_indices), dtype=bool)

        # Indices in 'rest_indices' that are close
        check_indices = np.where(potential_overlap_mask)[0]

        for idx in check_indices:
            iou = box_iou_3d_pair(current_box, rest_boxes[idx])
            if iou > iou_threshold:
                keep_mask[idx] = False

        order = order[1:][keep_mask]

    return keep


def box_encode(boxes, anchors):
    """
    Encodes Ground Truth boxes relative to Anchors.
    Args:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
        anchors: (N, 7) [x, y, z, w, l, h, yaw]
    Returns:
        targets: (N, 7) [dx, dy, dz, dw, dl, dh, dyaw]
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(boxes):
        boxes = torch.tensor(boxes)
    if not torch.is_tensor(anchors):
        anchors = torch.tensor(anchors)

    # Anchor diagonal for normalization
    d_a = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    # Center offsets
    dx = (boxes[:, 0] - anchors[:, 0]) / d_a
    dy = (boxes[:, 1] - anchors[:, 1]) / d_a
    dz = (boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]  # Normalizing by height

    # Dimension offsets (log space)
    # Add epsilon to prevent log(0)
    dw = torch.log(boxes[:, 3] / (anchors[:, 3] + 1e-6))
    dl = torch.log(boxes[:, 4] / (anchors[:, 4] + 1e-6))
    dh = torch.log(boxes[:, 5] / (anchors[:, 5] + 1e-6))

    # Yaw offset
    dyaw = boxes[:, 6] - anchors[:, 6]

    targets = torch.stack([dx, dy, dz, dw, dl, dh, dyaw], dim=1)
    return targets


def box_decode(reg, anchors):
    """
    Decodes Regression outputs to Absolute boxes.
    Args:
        reg: (N, 7) [dx, dy, dz, dw, dl, dh, dyaw]
        anchors: (N, 7) [x, y, z, w, l, h, yaw]
    Returns:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
    """
    if not torch.is_tensor(reg):
        reg = torch.tensor(reg)
    if not torch.is_tensor(anchors):
        anchors = torch.tensor(anchors)

    d_a = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    x = reg[:, 0] * d_a + anchors[:, 0]
    y = reg[:, 1] * d_a + anchors[:, 1]
    z = reg[:, 2] * anchors[:, 5] + anchors[:, 2]

    w = torch.exp(reg[:, 3]) * anchors[:, 3]
    l = torch.exp(reg[:, 4]) * anchors[:, 4]
    h = torch.exp(reg[:, 5]) * anchors[:, 5]

    yaw = reg[:, 6] + anchors[:, 6]

    boxes = torch.stack([x, y, z, w, l, h, yaw], dim=1)
    return boxes

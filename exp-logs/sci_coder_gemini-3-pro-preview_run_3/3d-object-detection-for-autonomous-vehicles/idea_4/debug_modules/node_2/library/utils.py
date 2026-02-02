import numpy as np
import torch
import cv2
import logging
import sys
import os
from library.config import Config


def setup_logger(log_file=None):
    """
    Sets up a logger that writes to both console and a file.
    """
    logger = logging.getLogger("PointPillars")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def encode_boxes(gt_boxes, anchors):
    """
    Encodes ground truth boxes into regression targets relative to anchors.

    Args:
        gt_boxes: (N, 7) Tensor [x, y, z, w, l, h, yaw]
        anchors: (N, 7) Tensor [x, y, z, w, l, h, yaw]

    Returns:
        targets: (N, 7) Tensor [dx, dy, dz, dw, dl, dh, dyaw]
    """
    # Anchors dimensions
    xa, ya, za = anchors[..., 0], anchors[..., 1], anchors[..., 2]
    wa, la, ha = anchors[..., 3], anchors[..., 4], anchors[..., 5]
    ra = anchors[..., 6]

    # GT dimensions
    xg, yg, zg = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2]
    wg, lg, hg = gt_boxes[..., 3], gt_boxes[..., 4], gt_boxes[..., 5]
    rg = gt_boxes[..., 6]

    # Diagonal of anchor base
    da = torch.sqrt(wa**2 + la**2)

    # Encoding
    tx = (xg - xa) / da
    ty = (yg - ya) / da
    tz = (zg - za) / ha
    tw = torch.log(wg / wa)
    tl = torch.log(lg / la)
    th = torch.log(hg / ha)
    tr = rg - ra

    targets = torch.stack([tx, ty, tz, tw, tl, th, tr], dim=-1)
    return targets


def decode_boxes(deltas, anchors):
    """
    Decodes regression deltas back to 3D boxes.

    Args:
        deltas: (N, 7) Tensor/Array [dx, dy, dz, dw, dl, dh, dyaw]
        anchors: (N, 7) Tensor/Array [x, y, z, w, l, h, yaw]

    Returns:
        boxes: (N, 7) Tensor [x, y, z, w, l, h, yaw]
    """
    is_numpy = False
    if isinstance(deltas, np.ndarray):
        deltas = torch.from_numpy(deltas)
        is_numpy = True
    if isinstance(anchors, np.ndarray):
        anchors = torch.from_numpy(anchors)

    xa, ya, za = anchors[..., 0], anchors[..., 1], anchors[..., 2]
    wa, la, ha = anchors[..., 3], anchors[..., 4], anchors[..., 5]
    ra = anchors[..., 6]

    dx, dy, dz = deltas[..., 0], deltas[..., 1], deltas[..., 2]
    dw, dl, dh = deltas[..., 3], deltas[..., 4], deltas[..., 5]
    dr = deltas[..., 6]

    da = torch.sqrt(wa**2 + la**2)

    xg = dx * da + xa
    yg = dy * da + ya
    zg = dz * ha + za
    wg = torch.exp(dw) * wa
    lg = torch.exp(dl) * la
    hg = torch.exp(dh) * ha
    rg = dr + ra

    boxes = torch.stack([xg, yg, zg, wg, lg, hg, rg], dim=-1)

    if is_numpy:
        return boxes.numpy()
    return boxes


def iou2d_nearest(anchors, gt_boxes):
    """
    Calculates 2D IoU assuming boxes are roughly axis-aligned.
    Used for efficient anchor matching during training.

    Args:
        anchors: (N, 7) Tensor
        gt_boxes: (M, 7) Tensor

    Returns:
        iou: (N, M) Tensor
    """
    if gt_boxes.shape[0] == 0:
        return torch.zeros((anchors.shape[0], 0), device=anchors.device)

    N = anchors.shape[0]
    M = gt_boxes.shape[0]

    # Extract parameters
    ax, ay, aw, al, ar = (
        anchors[:, 0],
        anchors[:, 1],
        anchors[:, 3],
        anchors[:, 4],
        anchors[:, 6],
    )
    gx, gy, gw, gl, gr = (
        gt_boxes[:, 0],
        gt_boxes[:, 1],
        gt_boxes[:, 3],
        gt_boxes[:, 4],
        gt_boxes[:, 6],
    )

    # Expand to (N, M)
    ax = ax.view(N, 1)
    ay = ay.view(N, 1)
    aw = aw.view(N, 1)
    al = al.view(N, 1)
    ar = ar.view(N, 1)

    gx = gx.view(1, M)
    gy = gy.view(1, M)
    gw = gw.view(1, M)
    gl = gl.view(1, M)
    gr = gr.view(1, M)

    # 1. Orientation Mismatch Check
    # Calculate angle difference modulo pi (180 degrees)
    # We treat boxes facing opposite directions as overlapping for IoU purposes
    angle_diff = torch.abs(ar - gr)
    angle_diff = angle_diff % np.pi
    angle_diff = torch.min(angle_diff, np.pi - angle_diff)

    # Only consider anchors that are roughly aligned with GT (within 45 degrees)
    orientation_mask = angle_diff < (np.pi / 4)

    # 2. Axis-Aligned Bounding Box (AABB) IoU
    # Calculate min/max coordinates
    a_min_x = ax - aw / 2
    a_min_y = ay - al / 2
    a_max_x = ax + aw / 2
    a_max_y = ay + al / 2

    g_min_x = gx - gw / 2
    g_min_y = gy - gl / 2
    g_max_x = gx + gw / 2
    g_max_y = gy + gl / 2

    # Intersection
    inter_min_x = torch.max(a_min_x, g_min_x)
    inter_min_y = torch.max(a_min_y, g_min_y)
    inter_max_x = torch.min(a_max_x, g_max_x)
    inter_max_y = torch.min(a_max_y, g_max_y)

    inter_w = torch.clamp(inter_max_x - inter_min_x, min=0)
    inter_h = torch.clamp(inter_max_y - inter_min_y, min=0)

    inter_area = inter_w * inter_h

    area_a = aw * al
    area_b = gw * gl

    union_area = area_a + area_b - inter_area

    iou = inter_area / (union_area + 1e-6)

    # Zero out IoU where orientation is mismatched
    iou = iou * orientation_mask.float()

    return iou


def polygon_intersection_area(box1, box2):
    """
    Calculates intersection area of two rotated rectangles using OpenCV.
    Args:
        box1, box2: [x, y, z, w, l, h, yaw]
    """
    # cv2.RotatedRect format: ((center_x, center_y), (width, height), angle_deg)
    # Note: Dataset yaw is in radians. CV2 angle is degrees.
    # We use degrees(yaw) directly.

    rect1 = (
        (float(box1[0]), float(box1[1])),
        (float(box1[3]), float(box1[4])),
        float(np.degrees(box1[6])),
    )
    rect2 = (
        (float(box2[0]), float(box2[1])),
        (float(box2[3]), float(box2[4])),
        float(np.degrees(box2[6])),
    )

    try:
        int_type, int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)

        if int_type == cv2.INTERSECT_NONE:
            return 0.0
        elif int_type == cv2.INTERSECT_FULL:
            # One is inside the other, return area of the smaller one
            area1 = box1[3] * box1[4]
            area2 = box2[3] * box2[4]
            return min(area1, area2)
        else:
            if int_pts is not None:
                # cv2 returns (N, 1, 2), needs (N, 2)
                # Compute area of the intersection polygon
                # convexHull ensures points are ordered
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                return cv2.contourArea(order_pts)
            return 0.0
    except Exception:
        return 0.0


def iou3d_cpu(boxes_a, boxes_b):
    """
    Calculates 3D IoU between two sets of boxes on CPU.
    Formula: (BEV_Intersection * Height_Overlap) / (Vol_A + Vol_B - Intersection_Vol)

    Args:
        boxes_a: (N, 7) numpy array
        boxes_b: (M, 7) numpy array

    Returns:
        iou: (N, M) numpy array
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            ba = boxes_a[i]
            bb = boxes_b[j]

            # 1. Height Overlap
            za_min = ba[2] - ba[5] / 2
            za_max = ba[2] + ba[5] / 2
            zb_min = bb[2] - bb[5] / 2
            zb_max = bb[2] + bb[5] / 2

            h_overlap = max(0, min(za_max, zb_max) - max(za_min, zb_min))

            if h_overlap == 0:
                continue

            # 2. Fast Distance Check (Optimization)
            # If centers are too far, skip expensive polygon intersection
            dist_sq = (ba[0] - bb[0]) ** 2 + (ba[1] - bb[1]) ** 2
            max_radius_sum = max(ba[3], ba[4]) + max(bb[3], bb[4])
            if dist_sq > max_radius_sum**2:
                continue

            # 3. BEV Intersection (Rotated)
            bev_area = polygon_intersection_area(ba, bb)

            if bev_area == 0:
                continue

            intersection_vol = bev_area * h_overlap

            vol_a = ba[3] * ba[4] * ba[5]
            vol_b = bb[3] * bb[4] * bb[5]

            union_vol = vol_a + vol_b - intersection_vol

            iou_matrix[i, j] = intersection_vol / union_vol if union_vol > 0 else 0.0

    return iou_matrix


def nms_3d(boxes, scores, threshold=0.1, max_detections=100):
    """
    Non-Maximum Suppression for 3D boxes using 3D IoU.

    Args:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
        scores: (N,)
        threshold: IoU threshold for suppression
        max_detections: Maximum number of boxes to keep

    Returns:
        keep_indices: list of indices to keep
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_detections:
            break

        if order.size == 1:
            break

        # Compare current box (i) with the rest
        rest_indices = order[1:]

        # Calculate IoU between box[i] and boxes[rest]
        current_box = boxes[i : i + 1]
        rest_boxes = boxes[rest_indices]

        # iou3d_cpu returns (1, M)
        ious = iou3d_cpu(current_box, rest_boxes)[0]

        # Keep boxes with IoU < threshold
        inds_to_keep = np.where(ious < threshold)[0]

        # Update order (inds_to_keep is relative to rest_indices)
        order = order[inds_to_keep + 1]

    return keep

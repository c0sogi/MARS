import numpy as np
import torch
from shapely.geometry import Polygon
from library.config import Config


def box3d_to_corners(boxes):
    """
    Convert 3D bounding boxes to 8 corners.

    Args:
        boxes: numpy array of shape (N, 7) [x, y, z, w, l, h, yaw]
               or (7,) for a single box.

    Returns:
        corners: numpy array of shape (N, 8, 3)
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    num_boxes = boxes.shape[0]

    # Unpack
    x = boxes[:, 0]
    y = boxes[:, 1]
    z = boxes[:, 2]
    w = boxes[:, 3]  # width (x-axis at yaw=0)
    l = boxes[:, 4]  # length (y-axis at yaw=0, forward/back)
    h = boxes[:, 5]  # height (z-axis)
    yaw = boxes[:, 6]

    # 1. Create template corners in local frame (centered at 0)
    # x_corners: [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    # y_corners: [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    # z_corners: [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    x_corners = w[:, np.newaxis] / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = l[:, np.newaxis] / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h[:, np.newaxis] / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # 2. Rotate
    # R = [[cos, -sin], [sin, cos]]
    c = np.cos(yaw)
    s = np.sin(yaw)

    # Shape: (N, 8)
    # x_rot = x_local * cos - y_local * sin
    # y_rot = x_local * sin + y_local * cos
    x_rot = x_corners * c[:, np.newaxis] - y_corners * s[:, np.newaxis]
    y_rot = x_corners * s[:, np.newaxis] + y_corners * c[:, np.newaxis]

    # 3. Translate
    x_final = x[:, np.newaxis] + x_rot
    y_final = y[:, np.newaxis] + y_rot
    z_final = z[:, np.newaxis] + z_corners

    # Stack: (N, 8, 3)
    corners = np.stack([x_final, y_final, z_final], axis=-1)

    return corners


def points_in_boxes_cpu(points, boxes):
    """
    Check points against boxes to see which points are inside which box.
    Used for GT Database generation and verification.

    Args:
        points: (N, 3) numpy array
        boxes: (M, 7) numpy array [x, y, z, w, l, h, yaw]

    Returns:
        point_indices: list of length M, where each element is an array of indices
                       of points inside that box.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    num_boxes = boxes.shape[0]
    point_indices = []

    for i in range(num_boxes):
        box = boxes[i]
        center = box[0:3]
        w, l, h = box[3], box[4], box[5]
        yaw = box[6]

        # Translate points to box center
        pts_local = points - center

        # Rotate points to align with box axes
        # We rotate points by -yaw to align them with axis-aligned box
        c = np.cos(-yaw)
        s = np.sin(-yaw)

        x_local = pts_local[:, 0] * c - pts_local[:, 1] * s
        y_local = pts_local[:, 0] * s + pts_local[:, 1] * c
        z_local = pts_local[:, 2]

        # Check bounds
        # width is along x, length is along y, height is along z
        in_x = np.abs(x_local) <= (w / 2)
        in_y = np.abs(y_local) <= (l / 2)
        in_z = np.abs(z_local) <= (h / 2)

        mask = in_x & in_y & in_z
        indices = np.where(mask)[0]
        point_indices.append(indices)

    return point_indices


def iou3d_shapely(boxes_a, boxes_b):
    """
    Calculate 3D IoU using Shapely for BEV intersection and height overlap.
    Metric: IoU = (Area_Inter * Height_Inter) / (Vol_A + Vol_B - (Area_Inter * Height_Inter))

    Args:
        boxes_a: (N, 7) numpy array
        boxes_b: (M, 7) numpy array

    Returns:
        iou: (N, M) numpy array
    """
    if isinstance(boxes_a, torch.Tensor):
        boxes_a = boxes_a.detach().cpu().numpy()
    if isinstance(boxes_b, torch.Tensor):
        boxes_b = boxes_b.detach().cpu().numpy()

    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    iou_matrix = np.zeros((N, M), dtype=np.float32)

    # Get corners for BEV polygons (bottom 4 corners)
    # box3d_to_corners returns order:
    # 0: (+w/2, +l/2, +h/2) -> Top Right Front (local)
    # ...
    # We need the projection on XY plane.
    # Let's just use the function and take x,y of the first 4 corners (top face) or last 4 (bottom face).
    # Actually, box3d_to_corners generates top 4 then bottom 4.
    # Indices 0,1,2,3 are top face. 4,5,6,7 are bottom face.
    # Let's use indices 0,1,2,3 for the polygon (projected to BEV).

    corners_a = box3d_to_corners(boxes_a)  # (N, 8, 3)
    corners_b = box3d_to_corners(boxes_b)  # (M, 8, 3)

    # Pre-compute volumes
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    # Pre-compute height ranges
    # z_center is at index 2, height at 5
    za_min = boxes_a[:, 2] - boxes_a[:, 5] / 2
    za_max = boxes_a[:, 2] + boxes_a[:, 5] / 2
    zb_min = boxes_b[:, 2] - boxes_b[:, 5] / 2
    zb_max = boxes_b[:, 2] + boxes_b[:, 5] / 2

    for i in range(N):
        # Create Polygon for box A
        # Using first 4 corners (x, y)
        poly_a_coords = corners_a[i, :4, :2]
        poly_a = Polygon(poly_a_coords)

        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)

        for j in range(M):
            # Height Intersection
            h_inter = min(za_max[i], zb_max[j]) - max(za_min[i], zb_min[j])

            if h_inter <= 0:
                iou_matrix[i, j] = 0.0
                continue

            # BEV Intersection
            poly_b_coords = corners_b[j, :4, :2]
            poly_b = Polygon(poly_b_coords)

            if not poly_b.is_valid:
                poly_b = poly_b.buffer(0)

            try:
                inter_area = poly_a.intersection(poly_b).area
            except Exception:
                inter_area = 0.0

            if inter_area <= 0:
                iou_matrix[i, j] = 0.0
                continue

            intersection_vol = inter_area * h_inter
            union_vol = vol_a[i] + vol_b[j] - intersection_vol

            if union_vol > 0:
                iou_matrix[i, j] = intersection_vol / union_vol
            else:
                iou_matrix[i, j] = 0.0

    return iou_matrix


def nms_3d(boxes, scores, iou_threshold=0.1):
    """
    Non-Maximum Suppression for 3D Bounding Boxes.

    Args:
        boxes: (N, 7) numpy array or tensor
        scores: (N,) numpy array or tensor
        iou_threshold: float

    Returns:
        keep_indices: list of indices to keep
    """
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = boxes

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = scores

    if len(boxes_np) == 0:
        return []

    # Sort by score descending
    order = np.argsort(scores_np)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compare this box with the rest
        # We compute IoU between box[i] and box[order[1:]]
        # Since calculating all IoUs is expensive, we can do a quick check?
        # For strict NMS, we should calculate 3D IoU.
        # To speed up, we can filter by BEV bounding rect first, but let's do direct IoU for correctness.

        rest_indices = order[1:]

        # Calculate IoU between box i and all remaining boxes
        # We can use a simplified loop or call iou3d_shapely (which is N*M).
        # Calling iou3d_shapely for 1 vs M is okay.

        current_box = boxes_np[i : i + 1]
        other_boxes = boxes_np[rest_indices]

        ious = iou3d_shapely(current_box, other_boxes)[0]  # Shape (M,)

        # Find indices where IoU < threshold (keep these)
        inds_to_keep = np.where(ious < iou_threshold)[0]

        # Update order (inds_to_keep are indices into rest_indices)
        order = rest_indices[inds_to_keep]

    return keep

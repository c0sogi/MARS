import torch
import numpy as np
import cv2
from library.config import Config


def limit_period(val, offset=0.5, period=np.pi):
    """
    Limits the value of an angle to the range [-period/2, period/2].
    """
    return val - torch.floor(val / period + offset) * period


def box_decode(box_encodings, anchors):
    """
    Decodes regression targets (residuals) back to absolute 3D bounding box coordinates.

    Args:
        box_encodings (torch.Tensor): Predicted residuals of shape (..., 7).
                                      [dx, dy, dz, dw, dl, dh, dyaw]
        anchors (torch.Tensor): Anchor/Proposal boxes of shape (..., 7).
                                [x, y, z, w, l, h, yaw]

    Returns:
        torch.Tensor: Decoded boxes of shape (..., 7).
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    diagonal = torch.sqrt(wa**2 + la**2)

    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    wg = torch.exp(wt) * wa
    lg = torch.exp(lt) * la
    hg = torch.exp(ht) * ha

    rg = rt + ra

    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


def box_encode(boxes, anchors):
    """
    Encodes ground truth boxes into regression targets relative to anchors/proposals.

    Args:
        boxes (torch.Tensor): Ground truth boxes of shape (..., 7).
        anchors (torch.Tensor): Anchor/Proposal boxes of shape (..., 7).

    Returns:
        torch.Tensor: Encoded residuals of shape (..., 7).
    """
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)

    diagonal = torch.sqrt(wa**2 + la**2)

    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    wt = torch.log(wg / wa)
    lt = torch.log(lg / la)
    ht = torch.log(hg / ha)

    rt = rg - ra

    return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)


def points_in_boxes_gpu(points, boxes):
    """
    Checks which points are contained within which 3D bounding boxes.
    Uses GPU broadcasting for efficiency.

    Args:
        points (torch.Tensor): Point cloud data of shape (N, 3) [x, y, z].
        boxes (torch.Tensor): Bounding boxes of shape (M, 7) [x, y, z, w, l, h, yaw].

    Returns:
        torch.Tensor: Boolean mask of shape (N, M) where mask[i, j] is True if point i is in box j.
    """
    # Ensure inputs are tensors
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)

    device = points.device
    boxes = boxes.to(device)

    N = points.shape[0]
    M = boxes.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.bool, device=device)

    # Expand dimensions for broadcasting: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    # Shift points to be relative to box centers
    box_centers = boxes[:, :3]
    shift = points[:, None, :3] - box_centers[None, :, :]

    # Extract box dimensions and angles
    # boxes: [x, y, z, w, l, h, yaw]
    # w: x-axis dim, l: y-axis dim (in box frame)
    box_dims = boxes[:, 3:6]
    box_yaw = boxes[:, 6]

    # Rotate points into box local coordinate system
    # Rotation matrix inverse (transpose) for yaw angle alpha:
    # [ cos(a)   sin(a)   0 ]
    # [ -sin(a)  cos(a)   0 ]
    # [ 0        0        1 ]

    cos_a = torch.cos(box_yaw)
    sin_a = torch.sin(box_yaw)

    # Perform rotation (only x and y change)
    dx = shift[:, :, 0]
    dy = shift[:, :, 1]
    dz = shift[:, :, 2]

    local_x = dx * cos_a[None, :] + dy * sin_a[None, :]
    local_y = -dx * sin_a[None, :] + dy * cos_a[None, :]
    local_z = dz

    # Check bounds
    # Local limits are [-w/2, w/2], [-l/2, l/2], [-h/2, h/2]
    w_half = box_dims[:, 0] / 2.0
    l_half = box_dims[:, 1] / 2.0
    h_half = box_dims[:, 2] / 2.0

    in_x = torch.abs(local_x) <= w_half[None, :]
    in_y = torch.abs(local_y) <= l_half[None, :]
    in_z = torch.abs(local_z) <= h_half[None, :]

    return in_x & in_y & in_z


def iou_bev(boxes_a, boxes_b):
    """
    Calculates the Bird's Eye View (BEV) Intersection over Union for rotated boxes.
    Uses OpenCV for polygon intersection on CPU.

    Args:
        boxes_a (torch.Tensor or np.ndarray): (N, 7) [x, y, z, w, l, h, yaw]
        boxes_b (torch.Tensor or np.ndarray): (M, 7) [x, y, z, w, l, h, yaw]

    Returns:
        torch.Tensor: (N, M) matrix of BEV IoU values.
    """
    # Convert to numpy for OpenCV
    if isinstance(boxes_a, torch.Tensor):
        boxes_a_np = boxes_a.detach().cpu().numpy()
    else:
        boxes_a_np = boxes_a

    if isinstance(boxes_b, torch.Tensor):
        boxes_b_np = boxes_b.detach().cpu().numpy()
    else:
        boxes_b_np = boxes_b

    N = boxes_a_np.shape[0]
    M = boxes_b_np.shape[0]

    iou_matrix = np.zeros((N, M), dtype=np.float32)

    # Pre-compute areas
    area_a = boxes_a_np[:, 3] * boxes_a_np[:, 4]
    area_b = boxes_b_np[:, 3] * boxes_b_np[:, 4]

    for i in range(N):
        # OpenCV RotatedRect format: ((center_x, center_y), (width, height), angle_deg)
        # Note: We map Box Length (l) to Height and Width (w) to Width in OpenCV terms,
        # or just consistent mapping.
        # Dataset Yaw: Counter-clockwise from X. OpenCV Angle: Clockwise.
        # We use -degrees(yaw).

        rect_a = (
            (float(boxes_a_np[i, 0]), float(boxes_a_np[i, 1])),
            (float(boxes_a_np[i, 3]), float(boxes_a_np[i, 4])),
            float(-np.degrees(boxes_a_np[i, 6])),
        )

        for j in range(M):
            rect_b = (
                (float(boxes_b_np[j, 0]), float(boxes_b_np[j, 1])),
                (float(boxes_b_np[j, 3]), float(boxes_b_np[j, 4])),
                float(-np.degrees(boxes_b_np[j, 6])),
            )

            try:
                # Calculate intersection
                int_type, int_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                if int_type != cv2.INTERSECT_NONE and int_pts is not None:
                    # int_pts is (K, 1, 2)
                    # Order vertices for area calculation? contourArea handles it.
                    intersection_area = cv2.contourArea(int_pts)
                else:
                    intersection_area = 0.0
            except:
                intersection_area = 0.0

            union_area = area_a[i] + area_b[j] - intersection_area
            if union_area > 1e-6:
                iou_matrix[i, j] = intersection_area / union_area

    return torch.tensor(iou_matrix)


def iou3d(boxes_a, boxes_b):
    """
    Calculates the 3D Intersection over Union.
    IoU = (Area_BEV_Intersection * Height_Intersection) / Union_Volume

    Args:
        boxes_a (torch.Tensor): (N, 7) [x, y, z, w, l, h, yaw]
        boxes_b (torch.Tensor): (M, 7) [x, y, z, w, l, h, yaw]

    Returns:
        torch.Tensor: (N, M) matrix of 3D IoU values.
    """
    device = (
        boxes_a.device if isinstance(boxes_a, torch.Tensor) else torch.device("cpu")
    )

    # 1. Calculate BEV IoU (Area Intersection / Area Union)
    # Note: We need the raw intersection area for 3D calc, but iou_bev returns ratio.
    # Let's reconstruct or modify.
    # To keep code clean, we calculate BEV Intersection Area separately or infer it.
    # Let's compute BEV Intersection Area explicitly here.

    # Convert to numpy for OpenCV
    if isinstance(boxes_a, torch.Tensor):
        boxes_a_np = boxes_a.detach().cpu().numpy()
    else:
        boxes_a_np = boxes_a

    if isinstance(boxes_b, torch.Tensor):
        boxes_b_np = boxes_b.detach().cpu().numpy()
    else:
        boxes_b_np = boxes_b

    N = boxes_a_np.shape[0]
    M = boxes_b_np.shape[0]

    bev_intersection = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        rect_a = (
            (float(boxes_a_np[i, 0]), float(boxes_a_np[i, 1])),
            (float(boxes_a_np[i, 3]), float(boxes_a_np[i, 4])),
            float(-np.degrees(boxes_a_np[i, 6])),
        )
        for j in range(M):
            rect_b = (
                (float(boxes_b_np[j, 0]), float(boxes_b_np[j, 1])),
                (float(boxes_b_np[j, 3]), float(boxes_b_np[j, 4])),
                float(-np.degrees(boxes_b_np[j, 6])),
            )
            try:
                int_type, int_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
                if int_type != cv2.INTERSECT_NONE and int_pts is not None:
                    bev_intersection[i, j] = cv2.contourArea(int_pts)
            except:
                pass

    bev_intersection = torch.tensor(bev_intersection, device=device)

    # 2. Calculate Height Intersection
    # z is center, h is height.
    # min_z = z - h/2, max_z = z + h/2
    za = boxes_a[:, 2]
    ha = boxes_a[:, 5]
    za_min = za - ha / 2
    za_max = za + ha / 2

    zb = boxes_b[:, 2]
    hb = boxes_b[:, 5]
    zb_min = zb - hb / 2
    zb_max = zb + hb / 2

    # Broadcast for (N, M)
    # max(min_a, min_b)
    inter_min = torch.max(za_min[:, None], zb_min[None, :])
    # min(max_a, max_b)
    inter_max = torch.min(za_max[:, None], zb_max[None, :])

    h_overlap = torch.clamp(inter_max - inter_min, min=0.0)

    # 3. Calculate 3D Intersection Volume
    intersection_vol = bev_intersection * h_overlap

    # 4. Calculate Union Volume
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    union_vol = vol_a[:, None] + vol_b[None, :] - intersection_vol

    iou = intersection_vol / torch.clamp(union_vol, min=1e-6)

    return iou


def nms_3d(boxes, scores, iou_threshold=0.1):
    """
    Performs Non-Maximum Suppression on 3D bounding boxes.
    Uses BEV IoU for overlap calculation.

    Args:
        boxes (torch.Tensor): (N, 7) [x, y, z, w, l, h, yaw]
        scores (torch.Tensor): (N,) Confidence scores
        iou_threshold (float): IoU threshold for suppression

    Returns:
        torch.Tensor: Indices of kept boxes.
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]

    # Compute IoU matrix (BEV is standard for NMS in 3D usually, or 3D IoU)
    # Using BEV IoU is safer/standard for driving datasets to avoid stacking
    iou_matrix = iou_bev(boxes, boxes).to(boxes.device)

    keep = []
    suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)

    for i in range(boxes.shape[0]):
        if suppressed[i]:
            continue

        keep.append(order[i])

        # Suppress all boxes with IoU > threshold
        # iou_matrix[i, j] > thresh
        overlap = iou_matrix[i] > iou_threshold

        # Don't suppress self (though loop logic handles it naturally as we move forward)
        # We only need to suppress indices > i
        overlap[: i + 1] = False

        suppressed = suppressed | overlap

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

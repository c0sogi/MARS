import numpy as np
import torch
import cv2


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: float array of shape (..., 4) with (w, x, y, z)
    Returns:
        matrices: float array of shape (..., 3, 3)
    """
    is_torch = isinstance(quaternions, torch.Tensor)
    if is_torch:
        q = quaternions.detach().cpu().numpy()
    else:
        q = quaternions

    # Normalize quaternion
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / (q_norm + 1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    row0 = np.stack([1 - 2 * y2 - 2 * z2, 2 * xy - 2 * zw, 2 * xz + 2 * yw], axis=-1)
    row1 = np.stack([2 * xy + 2 * zw, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * xw], axis=-1)
    row2 = np.stack([2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * x2 - 2 * y2], axis=-1)

    matrix = np.stack([row0, row1, row2], axis=-2)

    if is_torch:
        return torch.from_numpy(matrix).to(quaternions.device).type(quaternions.dtype)
    return matrix


def transform_points(points, trans, rot_quat=None, rot_mat=None, inverse=False):
    """
    Apply rigid transformation to points.
    Args:
        points: (N, 3)
        trans: (3,) translation vector
        rot_quat: (4,) quaternion (w, x, y, z)
        rot_mat: (3, 3) rotation matrix
        inverse: if True, apply inverse transform
    """
    is_torch = isinstance(points, torch.Tensor)

    if rot_mat is None and rot_quat is not None:
        rot_mat = quaternion_to_matrix(rot_quat)

    if is_torch:
        if not isinstance(rot_mat, torch.Tensor):
            rot_mat = torch.from_numpy(rot_mat).to(points.device).type(points.dtype)
        if not isinstance(trans, torch.Tensor):
            trans = torch.from_numpy(trans).to(points.device).type(points.dtype)

    if inverse:
        # Inverse: p = R^T (p_in - t)
        if is_torch:
            points_out = points - trans
            points_out = torch.matmul(
                points_out, rot_mat
            )  # Equivalent to p @ R which is p @ (R^T)^T
        else:
            points_out = points - trans
            points_out = np.dot(points_out, rot_mat)
    else:
        # Forward: p' = p R^T + t (assuming row vectors)
        if is_torch:
            points_out = torch.matmul(points, rot_mat.t()) + trans
        else:
            points_out = np.dot(points, rot_mat.T) + trans

    return points_out


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create a boolean mask indicating padded positions.
    Args:
        actual_num: (B,) tensor of actual lengths
        max_num: int, maximum length
    Returns:
        mask: (B, max_num) boolean tensor, True where padded (invalid)
    """
    actual_num = actual_num.unsqueeze(axis)
    arange = torch.arange(max_num, device=actual_num.device).unsqueeze(0)
    return arange >= actual_num


def box3d_to_corners(boxes):
    """
    Convert 3D boxes to 8 corners.
    Args:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
    Returns:
        corners: (N, 8, 3)
    """
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = boxes

    if boxes_np.shape[0] == 0:
        if is_torch:
            return torch.zeros((0, 8, 3), device=boxes.device, dtype=boxes.dtype)
        return np.zeros((0, 8, 3), dtype=np.float32)

    x, y, z = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2]
    w, l, h = boxes_np[:, 3], boxes_np[:, 4], boxes_np[:, 5]
    yaw = boxes_np[:, 6]

    # Canonical corners relative to center
    x_corners = w[:, None] / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = l[:, None] / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h[:, None] / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    # Rotation (Yaw around Z)
    c = np.cos(yaw)
    s = np.sin(yaw)

    x_new = corners[..., 0] * c[:, None] - corners[..., 1] * s[:, None]
    y_new = corners[..., 0] * s[:, None] + corners[..., 1] * c[:, None]

    corners[..., 0] = x_new
    corners[..., 1] = y_new

    # Translation
    corners[..., 0] += x[:, None]
    corners[..., 1] += y[:, None]
    corners[..., 2] += z[:, None]

    if is_torch:
        return torch.from_numpy(corners).to(boxes.device).type(boxes.dtype)
    return corners


def iou3d_global(boxes_a, boxes_b):
    """
    Calculate 3D IoU between two sets of boxes.
    Args:
        boxes_a: (N, 7) [x, y, z, w, l, h, yaw]
        boxes_b: (M, 7)
    Returns:
        iou: (N, M)
    """
    is_torch = isinstance(boxes_a, torch.Tensor)
    if is_torch:
        ba = boxes_a.detach().cpu().numpy()
        bb = boxes_b.detach().cpu().numpy()
    else:
        ba = boxes_a
        bb = boxes_b

    N = ba.shape[0]
    M = bb.shape[0]

    if N == 0 or M == 0:
        if is_torch:
            return torch.zeros((N, M), device=boxes_a.device)
        return np.zeros((N, M))

    # Height Overlap
    az_min = ba[:, 2] - ba[:, 5] / 2
    az_max = ba[:, 2] + ba[:, 5] / 2
    bz_min = bb[:, 2] - bb[:, 5] / 2
    bz_max = bb[:, 2] + bb[:, 5] / 2

    inter_h_min = np.maximum(az_min[:, None], bz_min[None, :])
    inter_h_max = np.minimum(az_max[:, None], bz_max[None, :])
    inter_h = np.maximum(0.0, inter_h_max - inter_h_min)

    inter_area = np.zeros((N, M), dtype=np.float32)

    # 2D Intersection Loop
    for i in range(N):
        # OpenCV RotatedRect: ((x, y), (w, h), angle_deg_cw)
        # Yaw is CCW radians. angle_deg = -degrees(yaw)
        rect_a = ((ba[i, 0], ba[i, 1]), (ba[i, 3], ba[i, 4]), -np.degrees(ba[i, 6]))

        for j in range(M):
            if inter_h[i, j] == 0:
                continue

            # Quick bounding circle check
            dist_sq = (ba[i, 0] - bb[j, 0]) ** 2 + (ba[i, 1] - bb[j, 1]) ** 2
            radius_sum = (
                np.sqrt(ba[i, 3] ** 2 + ba[i, 4] ** 2)
                + np.sqrt(bb[j, 3] ** 2 + bb[j, 4] ** 2)
            ) / 2
            if dist_sq > (radius_sum * 1.5) ** 2:  # 1.5 safety factor
                continue

            rect_b = ((bb[j, 0], bb[j, 1]), (bb[j, 3], bb[j, 4]), -np.degrees(bb[j, 6]))

            try:
                ret, vertices = cv2.rotatedRectangleIntersection(rect_a, rect_b)
                if ret != cv2.INTERSECT_NONE and vertices is not None:
                    inter_area[i, j] = cv2.contourArea(vertices)
            except:
                pass

    inter_vol = inter_area * inter_h

    vol_a = ba[:, 3] * ba[:, 4] * ba[:, 5]
    vol_b = bb[:, 3] * bb[:, 4] * bb[:, 5]

    union_vol = vol_a[:, None] + vol_b[None, :] - inter_vol

    iou = inter_vol / (union_vol + 1e-8)

    if is_torch:
        return torch.from_numpy(iou).to(boxes_a.device)
    return iou

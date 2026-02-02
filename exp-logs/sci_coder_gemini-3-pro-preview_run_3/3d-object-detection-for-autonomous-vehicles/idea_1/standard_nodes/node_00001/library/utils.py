import numpy as np
import torch
from library.config import Config


def read_points(file_path, dim=4):
    """
    Reads LIDAR point cloud data from a binary file.

    Args:
        file_path (str): Path to the .bin file.
        dim (int): Number of features per point (default: 4 for x, y, z, intensity).

    Returns:
        np.ndarray: Array of shape (N, dim) containing point cloud data.
    """
    try:
        points = np.fromfile(file_path, dtype=np.float32)

        # Handle 5-dim NuScenes format (x, y, z, intensity, ring_index)
        if points.size % 5 == 0:
            points = points.reshape(-1, 5)
            # Slice to keep only the requested dimensions (e.g., first 4)
            points = points[:, :dim]
        else:
            points = points.reshape(-1, dim)

        return points
    except FileNotFoundError:
        print(f"Error: Lidar file not found at {file_path}")
        return np.zeros((0, dim), dtype=np.float32)
    except ValueError:
        print(
            f"Error: File at {file_path} cannot be reshaped to dim {dim}. Size: {points.size}"
        )
        return np.zeros((0, dim), dtype=np.float32)


def limit_period(val, offset=0.5, period=np.pi):
    """
    Limits the value (usually angle) to a specific period.
    For example, to [-pi, pi].

    Args:
        val (torch.Tensor or np.ndarray): Input values.
        offset (float): Offset fraction of period.
        period (float): The period length (e.g., 2*pi).

    Returns:
        Result with values in [ -offset*period, (1-offset)*period ].
    """
    if isinstance(val, torch.Tensor):
        return val - torch.floor(val / period + offset) * period
    return val - np.floor(val / period + offset) * period


def iou_2d(boxes_a, boxes_b):
    """
    Calculate Axis-Aligned 2D Intersection over Union (IoU) between two sets of boxes.
    Used for anchor matching.

    Args:
        boxes_a (torch.Tensor): Shape (N, 4) [x, y, w, l]
        boxes_b (torch.Tensor): Shape (M, 4) [x, y, w, l]

    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    # Convert center-size to min-max coordinates
    # x_min, y_min, x_max, y_max

    # boxes_a: [N, 4]
    xa_min = boxes_a[:, 0] - boxes_a[:, 2] / 2
    ya_min = boxes_a[:, 1] - boxes_a[:, 3] / 2
    xa_max = boxes_a[:, 0] + boxes_a[:, 2] / 2
    ya_max = boxes_a[:, 1] + boxes_a[:, 3] / 2

    # boxes_b: [M, 4]
    xb_min = boxes_b[:, 0] - boxes_b[:, 2] / 2
    yb_min = boxes_b[:, 1] - boxes_b[:, 3] / 2
    xb_max = boxes_b[:, 0] + boxes_b[:, 2] / 2
    yb_max = boxes_b[:, 1] + boxes_b[:, 3] / 2

    # Expand dims for broadcasting
    # (N, 1) vs (1, M) -> (N, M)

    # Intersection coordinates
    inter_xmin = torch.max(xa_min.unsqueeze(1), xb_min.unsqueeze(0))
    inter_ymin = torch.max(ya_min.unsqueeze(1), yb_min.unsqueeze(0))
    inter_xmax = torch.min(xa_max.unsqueeze(1), xb_max.unsqueeze(0))
    inter_ymax = torch.min(ya_max.unsqueeze(1), yb_max.unsqueeze(0))

    # Intersection dimensions
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)

    intersection = inter_w * inter_h

    # Areas
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]

    # Union
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - intersection

    # IoU
    # Add epsilon to avoid division by zero
    iou = intersection / (union + 1e-6)

    return iou


def iou_3d(boxes_a, boxes_b):
    """
    Calculate Axis-Aligned 3D IoU.
    Metric definition: IoU = (Intersection 2D * Intersection Height) / Union 3D

    Args:
        boxes_a (torch.Tensor): Shape (N, 6) [x, y, z, w, l, h]
        boxes_b (torch.Tensor): Shape (M, 6) [x, y, z, w, l, h]

    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    # 1. Calculate 2D Intersection (BEV)
    # boxes: x, y, z, w, l, h
    # indices: 0, 1, 2, 3, 4, 5

    # 2D part (x, y, w, l)
    xa_min = boxes_a[:, 0] - boxes_a[:, 3] / 2
    ya_min = boxes_a[:, 1] - boxes_a[:, 4] / 2
    xa_max = boxes_a[:, 0] + boxes_a[:, 3] / 2
    ya_max = boxes_a[:, 1] + boxes_a[:, 4] / 2

    xb_min = boxes_b[:, 0] - boxes_b[:, 3] / 2
    yb_min = boxes_b[:, 1] - boxes_b[:, 4] / 2
    xb_max = boxes_b[:, 0] + boxes_b[:, 3] / 2
    yb_max = boxes_b[:, 1] + boxes_b[:, 4] / 2

    inter_xmin = torch.max(xa_min.unsqueeze(1), xb_min.unsqueeze(0))
    inter_ymin = torch.max(ya_min.unsqueeze(1), yb_min.unsqueeze(0))
    inter_xmax = torch.min(xa_max.unsqueeze(1), xb_max.unsqueeze(0))
    inter_ymax = torch.min(ya_max.unsqueeze(1), yb_max.unsqueeze(0))

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_l = torch.clamp(inter_ymax - inter_ymin, min=0)

    inter_area_2d = inter_w * inter_l

    # 2. Calculate Height Intersection
    za_min = boxes_a[:, 2] - boxes_a[:, 5] / 2
    za_max = boxes_a[:, 2] + boxes_a[:, 5] / 2

    zb_min = boxes_b[:, 2] - boxes_b[:, 5] / 2
    zb_max = boxes_b[:, 2] + boxes_b[:, 5] / 2

    inter_zmin = torch.max(za_min.unsqueeze(1), zb_min.unsqueeze(0))
    inter_zmax = torch.min(za_max.unsqueeze(1), zb_max.unsqueeze(0))

    inter_h = torch.clamp(inter_zmax - inter_zmin, min=0)

    # 3. 3D Intersection Volume
    inter_vol = inter_area_2d * inter_h

    # 4. 3D Union Volume
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    union_vol = vol_a.unsqueeze(1) + vol_b.unsqueeze(0) - inter_vol

    iou = inter_vol / (union_vol + 1e-6)

    return iou


def bbox3d2corners(bboxes):
    """
    Convert 3D bounding boxes to 8 corners.

    Args:
        bboxes (torch.Tensor): Shape (N, 7) [x, y, z, w, l, h, yaw]

    Returns:
        torch.Tensor: Shape (N, 8, 3)
    """
    centers = bboxes[:, 0:3]
    dims = bboxes[:, 3:6]
    yaw = bboxes[:, 6]

    # Reference box corners centered at origin (N, 8, 3)
    # x: l, y: w, z: h (depending on coord system, usually x=forward=length, y=left=width)
    # Config says: width, length, height.
    # Usually in Lidar coords: x forward, y left.
    # So length is along x, width is along y.
    # dims: [w, l, h] -> x_dim=l, y_dim=w, z_dim=h

    l = dims[:, 1]
    w = dims[:, 0]
    h = dims[:, 2]

    # Create corners relative to center
    # Front-Left-Top, Front-Right-Top, ...
    # x signs: 1, 1, -1, -1, 1, 1, -1, -1
    # y signs: 1, -1, -1, 1, 1, -1, -1, 1
    # z signs: 1, 1, 1, 1, -1, -1, -1, -1

    x_corners = (
        l.unsqueeze(1)
        / 2
        * torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=bboxes.device).type_as(
            bboxes
        )
    )
    y_corners = (
        w.unsqueeze(1)
        / 2
        * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1], device=bboxes.device).type_as(
            bboxes
        )
    )
    z_corners = (
        h.unsqueeze(1)
        / 2
        * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=bboxes.device).type_as(
            bboxes
        )
    )

    # Rotate
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # Rotation matrix around Z
    # x_new = x*cos - y*sin
    # y_new = x*sin + y*cos

    # Expand for broadcasting (N, 8)
    c = c.unsqueeze(1)
    s = s.unsqueeze(1)

    x_rot = x_corners * c - y_corners * s
    y_rot = x_corners * s + y_corners * c
    z_rot = z_corners

    # Translate
    corners = torch.stack([x_rot, y_rot, z_rot], dim=-1)  # (N, 8, 3)
    corners = corners + centers.unsqueeze(1)

    return corners

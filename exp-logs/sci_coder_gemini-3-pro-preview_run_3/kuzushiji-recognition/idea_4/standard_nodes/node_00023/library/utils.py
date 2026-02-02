import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Computes the gaussian radius for a given bounding box size and minimum overlap.
    Based on the CenterNet paper/implementation (CornerNet logic).

    Args:
        det_size (tuple): (height, width) of the bounding box.
        min_overlap (float): Minimum IoU overlap.

    Returns:
        float: The calculated radius.
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


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D gaussian on the heatmap array in-place.

    Args:
        heatmap (np.ndarray): The heatmap to update (H, W).
        center (tuple): (x, y) integer coordinates.
        radius (float): The gaussian radius (sigma).
        k (float): Scaling factor (usually 1 for ground truth).

    Returns:
        np.ndarray: The updated heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

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


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D gaussian kernel.

    Args:
        shape (tuple): (height, width) of the kernel.
        sigma (float): Standard deviation.

    Returns:
        np.ndarray: The 2D gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def xywh_to_xyxy(boxes):
    """
    Converts bounding boxes from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        boxes (torch.Tensor or np.ndarray): Shape (N, 4).

    Returns:
        Converted boxes with same type and shape.
    """
    if isinstance(boxes, torch.Tensor):
        x = boxes[..., 0]
        y = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        x = boxes[..., 0]
        y = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_xywh(boxes):
    """
    Converts bounding boxes from [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        boxes (torch.Tensor or np.ndarray): Shape (N, 4).

    Returns:
        Converted boxes with same type and shape.
    """
    if isinstance(boxes, torch.Tensor):
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        w = x2 - x1
        h = y2 - y1

        return torch.stack([x1, y1, w, h], dim=-1)
    else:
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        w = x2 - x1
        h = y2 - y1

        return np.stack([x1, y1, w, h], axis=-1)


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two sets of boxes.
    Expects boxes in [x1, y1, x2, y2] format.

    Args:
        box1 (torch.Tensor): Shape (N, 4).
        box2 (torch.Tensor): Shape (M, 4).

    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
    """
    # Ensure inputs are at least 2D
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt
    wh[wh < 0] = 0  # Clip negative widths/heights (no intersection)

    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    area1 = area1.unsqueeze(1).expand_as(inter)
    area2 = area2.unsqueeze(0).expand_as(inter)

    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    return iou

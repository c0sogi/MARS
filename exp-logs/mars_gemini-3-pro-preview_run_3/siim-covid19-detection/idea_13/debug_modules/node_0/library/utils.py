import os
import random
import numpy as np
import torch
from typing import List, Union, Tuple


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ====================================================
# Bounding Box Operations
# ====================================================


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) is the center of the box.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes IoU between two sets of boxes.
    boxes1: (N, 4) in (x1, y1, x2, y2)
    boxes2: (M, 4) in (x1, y1, x2, y2)

    Returns:
        iou: (N, M) matrix
        union: (N, M) matrix of union areas
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format.
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2).
    """
    # Assert boxes are valid (x2 >= x1 and y2 >= y1)
    # Note: We relax this assertion for empty sets or during training instabilities,
    # but ideally boxes should be well-formed.

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# ====================================================
# Submission Formatting
# ====================================================


def format_prediction_string(
    labels: List[str], boxes: Union[List, np.ndarray], scores: List[float]
) -> str:
    """
    Formats the predictions into the required submission string format.
    Format: "label score xmin ymin xmax ymax label score ..."

    Args:
        labels: List of class labels (e.g., ['opacity', 'opacity']).
        boxes: List of bounding boxes, each being [xmin, ymin, xmax, ymax].
        scores: List of confidence scores.

    Returns:
        A single string containing all predictions for an image/study.
    """
    pred_strings = []
    for label, score, box in zip(labels, scores, boxes):
        # Format box coordinates to reasonable precision
        # Box is expected to be xmin, ymin, xmax, ymax
        coords = [f"{float(c):.4f}" for c in box]
        pred_strings.append(f"{label} {float(score):.6f} {' '.join(coords)}")

    if not pred_strings:
        # If no predictions, return the default "none" string
        return "none 1 0 0 1 1"

    return " ".join(pred_strings)

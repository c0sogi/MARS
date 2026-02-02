import os
import random
import logging
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


def get_logger(filename):
    """
    Initializes and returns a logger that outputs to both console and file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(filename, mode="w")

        # Create formatters
        c_format = logging.Formatter("%(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def mask_to_boxes(mask, threshold=0.5):
    """
    Converts a binary segmentation mask into a list of bounding boxes.

    Args:
        mask (np.array): Probability map or binary mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(c)

        # Filter out extremely small noise if necessary (e.g., 1px dots)
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])

    return boxes


def compute_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at a specific IoU threshold.
    Follows the PASCAL VOC 2010 standard (all-point interpolation).

    Args:
        pred_boxes (list): List of dictionaries for each image.
                           Format: {'boxes': [[x1, y1, x2, y2], ...], 'scores': [0.9, 0.5, ...]}
        true_boxes (list): List of dictionaries for each image.
                           Format: {'boxes': [[x1, y1, x2, y2], ...]}
        iou_threshold (float): IoU threshold for a match to be considered a True Positive.

    Returns:
        float: The Average Precision (AP) for the dataset.
    """
    all_preds = []
    total_gt = 0

    # 1. Flatten all predictions and count total ground truths
    for i in range(len(pred_boxes)):
        preds = pred_boxes[i]
        gt = true_boxes[i]

        total_gt += len(gt["boxes"])

        for j in range(len(preds["boxes"])):
            all_preds.append(
                {
                    "box": preds["boxes"][j],
                    "score": preds["scores"][j],
                    "image_idx": i,
                    "matched": False,
                }
            )

    # If no ground truth boxes exist in the entire set, and we predicted nothing, AP is 0 (or undefined).
    # If we predicted something, it's False Positives.
    if total_gt == 0:
        return 0.0

    if not all_preds:
        return 0.0

    # 2. Sort predictions by confidence score (descending)
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track which GT boxes have been matched to avoid double counting
    matched_gt = set()

    # 3. Match predictions to ground truth
    for i, pred in enumerate(all_preds):
        img_idx = pred["image_idx"]
        pred_box = pred["box"]

        gt_boxes = true_boxes[img_idx]["boxes"]

        best_iou = 0
        best_gt_idx = -1

        # Find best matching GT box
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Determine TP/FP
        if best_iou > iou_threshold:
            if (img_idx, best_gt_idx) not in matched_gt:
                tp[i] = 1
                matched_gt.add((img_idx, best_gt_idx))
            else:
                fp[i] = 1  # Duplicate detection for same object
        else:
            fp[i] = 1

    # 4. Compute Precision and Recall curves
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 5. PASCAL VOC 2010 AP Calculation (All-point interpolation)
    # Add sentinel values to beginning and end
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # Integrate area under the curve
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    # Sum (Recall_delta * Precision_height)
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap

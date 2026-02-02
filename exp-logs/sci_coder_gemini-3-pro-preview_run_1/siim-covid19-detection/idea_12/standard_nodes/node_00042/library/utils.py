import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def bbox2mask(bboxes, height, width):
    """
    Converts a list of bounding boxes to a binary mask.

    Args:
        bboxes (list): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        np.ndarray: Binary mask of shape (height, width) with 1s inside boxes and 0s elsewhere.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        # Clip coordinates to image boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def mask2bbox(mask):
    """
    Converts a binary mask to a list of bounding boxes.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        list: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Only keep valid boxes
        if w > 0 and h > 0:
            bboxes.append([x, y, x + w, y + h])
    return bboxes


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
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


def get_score(pred_rows, gt_rows, iou_threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at a specific IoU threshold.
    Follows the PASCAL VOC 2010 metric (all-point interpolation).

    Args:
        pred_rows (list): List of dicts for each image.
                          Each dict contains 'boxes' (list of lists) and 'scores' (list of floats).
        gt_rows (list): List of dicts for each image.
                        Each dict contains 'boxes' (list of lists).
        iou_threshold (float): IoU threshold for a positive match.

    Returns:
        float: The Average Precision (AP) score.
    """
    # Flatten all predictions across all images
    all_preds = []
    for i, row in enumerate(pred_rows):
        boxes = row.get("boxes", [])
        scores = row.get("scores", [])
        for b, s in zip(boxes, scores):
            all_preds.append({"bbox": b, "score": s, "img_idx": i})

    # Sort predictions by confidence score in descending order
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    # Initialize True Positives and False Positives arrays
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track which ground truth boxes have been matched
    # gt_matched[img_idx][gt_box_idx] = boolean
    gt_matched = {
        i: [False] * len(gt_rows[i].get("boxes", [])) for i in range(len(gt_rows))
    }

    for i, pred in enumerate(all_preds):
        img_idx = pred["img_idx"]
        pred_box = pred["bbox"]

        gt_boxes = gt_rows[img_idx].get("boxes", [])

        best_iou = 0.0
        best_gt_idx = -1

        # Find the ground truth box with the highest IoU
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Determine if it's a TP or FP
        if best_iou > iou_threshold:
            if not gt_matched[img_idx][best_gt_idx]:
                tp[i] = 1.0
                gt_matched[img_idx][best_gt_idx] = True
            else:
                fp[i] = 1.0  # Matched a GT that was already matched (duplicate)
        else:
            fp[i] = 1.0  # IoU too low or no GT

    # Calculate Precision and Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    total_gt = sum([len(r.get("boxes", [])) for r in gt_rows])

    if total_gt == 0:
        return 0.0

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # PASCAL VOC 2010 Average Precision Calculation (All-point interpolation)
    # Prepend 0 to recalls and precisions for integration
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate Area Under Curve
    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of rectangular areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)

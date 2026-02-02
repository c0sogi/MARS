import os
import random
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
    Used for tracking losses and metrics during training.
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


def mask2bbox(mask, threshold=0.5):
    """
    Converts a probability mask into bounding boxes using contour detection.

    Args:
        mask (np.array): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: A list of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    # Binarize mask
    mask_bin = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        # Get bounding rect: x, y, w, h
        x, y, w, h = cv2.boundingRect(c)

        # Filter extremely small artifacts if necessary (optional, but good practice)
        if w < 2 or h < 2:
            continue

        # Convert to x_min, y_min, x_max, y_max
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) ndarray [x1, y1, x2, y2]
        boxes2: (M, 4) ndarray [x1, y1, x2, y2]

    Returns:
        iou: (N, M) ndarray
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Broadcasting for intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.maximum(0, rb - lt)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    # Avoid division by zero
    union = np.maximum(union, 1e-6)

    iou = inter / union
    return iou


def calculate_map(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    num_classes=1,
    iou_threshold=0.5,
):
    """
    Calculates Mean Average Precision (mAP) at a specific IoU threshold.
    Uses the 11-point interpolation or all-point interpolation (standard VOC).
    Here we implement all-point interpolation (VOC 2010+).

    Args:
        pred_boxes (list of np.array): List of predicted boxes for each image.
        pred_scores (list of np.array): List of scores for each image.
        pred_labels (list of np.array): List of labels for each image.
        gt_boxes (list of np.array): List of ground truth boxes for each image.
        gt_labels (list of np.array): List of ground truth labels for each image.
        num_classes (int): Number of classes.
        iou_threshold (float): IoU threshold for a match.

    Returns:
        float: mAP score.
    """
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Flatten data for the specific class
        total_gt = 0

        for i in range(len(gt_boxes)):
            # Get GT for this image and class
            if len(gt_labels[i]) > 0:
                mask_gt = gt_labels[i] == c
                gts = gt_boxes[i][mask_gt]
            else:
                gts = np.empty((0, 4))

            total_gt += len(gts)

            # Get Predictions for this image and class
            if len(pred_labels[i]) > 0:
                mask_pred = pred_labels[i] == c
                preds = pred_boxes[i][mask_pred]
                scores = pred_scores[i][mask_pred]
            else:
                preds = np.empty((0, 4))
                scores = np.array([])

            for j in range(len(preds)):
                detections.append([scores[j], i, preds[j]])  # score, image_idx, box

            ground_truths.append(gts)  # Store GTs per image to match by index

        if total_gt == 0:
            if len(detections) == 0:
                average_precisions.append(1.0)  # No GT, No Pred -> Perfect
            else:
                average_precisions.append(0.0)  # No GT, but Preds -> False Positives
            continue

        # Sort detections by score descending
        detections.sort(key=lambda x: x[0], reverse=True)

        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        # Keep track of which GTs have been matched
        gt_matched = [np.zeros(len(gts)) for gts in ground_truths]

        for d_idx, detection in enumerate(detections):
            score, img_idx, box = detection
            gts = ground_truths[img_idx]

            best_iou = 0
            best_gt_idx = -1

            if len(gts) > 0:
                # Calculate IoU with all GTs in this image
                # box shape (4,), gts shape (M, 4)
                # Expand box to (1, 4) for vectorization
                ious = box_iou(np.expand_dims(box, 0), gts)[0]  # returns shape (M,)
                best_iou = np.max(ious)
                best_gt_idx = np.argmax(ious)

            if best_iou >= iou_threshold:
                if gt_matched[img_idx][best_gt_idx] == 0:
                    TP[d_idx] = 1
                    gt_matched[img_idx][best_gt_idx] = 1
                else:
                    FP[d_idx] = 1  # Already matched (duplicate detection)
            else:
                FP[d_idx] = 1

        # Compute cumulative precision and recall
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        recalls = acc_TP / total_gt
        precisions = acc_TP / (acc_TP + acc_FP)

        # Compute AP using All-Point Interpolation (VOC 2010+)
        # Insert 0 at beginning and end for integration
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Integrate area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        average_precisions.append(ap)

    return np.mean(average_precisions)

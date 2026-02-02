import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list/array): [xmin, ymin, xmax, ymax]
        box2 (list/array): [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU value between 0.0 and 1.0
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def get_box_from_mask(mask, threshold=0.5, original_shape=None):
    """
    Extract bounding boxes from a probability mask using contour detection.

    Args:
        mask (np.array): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.
        original_shape (tuple, optional): (height, width) of the original image.
                                          If provided, boxes are rescaled to this size.

    Returns:
        list: List of bounding boxes in [xmin, ymin, xmax, ymax] format.
        list: List of confidence scores (mean probability within the box).
    """
    # Ensure mask is on CPU and numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask_h, mask_w = mask.shape

    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    scores = []

    for contour in contours:
        # Get bounding rect: x, y, w, h
        x, y, w, h = cv2.boundingRect(contour)

        # Filter very small artifacts (e.g., < 10 pixels area)
        if w * h < 10:
            continue

        xmin, ymin = x, y
        xmax, ymax = x + w, y + h

        # Calculate score before rescaling
        # Extract region of interest from the probability mask
        mask_roi = mask[ymin:ymax, xmin:xmax]
        binary_roi = binary_mask[ymin:ymax, xmin:xmax]

        # Calculate mean score of the pixels that are part of the mask
        if np.sum(binary_roi) > 0:
            score = np.sum(mask_roi * binary_roi) / np.sum(binary_roi)
        else:
            score = np.mean(mask_roi)

        # Rescale boxes if original shape is provided
        if original_shape is not None:
            orig_h, orig_w = original_shape
            scale_x = orig_w / mask_w
            scale_y = orig_h / mask_h

            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)

        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(float(score))

    return boxes, scores


def format_prediction_string(labels, confidences, boxes):
    """
    Format predictions into the competition submission string format.

    Args:
        labels (list): List of class labels (strings).
        confidences (list): List of confidence scores (floats).
        boxes (list): List of boxes, where each box is [xmin, ymin, xmax, ymax].
                      For study-level or 'none' predictions, boxes should be [0, 0, 1, 1].

    Returns:
        str: Prediction string "label confidence xmin ymin xmax ymax ..."
    """
    pred_strings = []

    for i in range(len(labels)):
        label = labels[i]
        conf = confidences[i]

        if boxes is None or (i < len(boxes) and boxes[i] is None):
            # Default 1-pixel box
            box_str = "0 0 1 1"
        elif i < len(boxes):
            b = boxes[i]
            # Ensure box format is correct
            if len(b) == 4:
                box_str = f"{b[0]} {b[1]} {b[2]} {b[3]}"
            else:
                box_str = "0 0 1 1"
        else:
            box_str = "0 0 1 1"

        pred_strings.append(f"{label} {conf} {box_str}")

    return " ".join(pred_strings)


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.5]):
    """
    Calculate Mean Average Precision (mAP) at specific IoU thresholds.

    Args:
        boxes_true (list): List of ground truth boxes [[x1,y1,x2,y2], ...].
        boxes_pred (list): List of predicted boxes [[x1,y1,x2,y2], ...].
        scores (list): Confidence scores for predicted boxes.
        thresholds (list): List of IoU thresholds to evaluate.

    Returns:
        float: mAP score averaged over thresholds.
    """
    if len(boxes_pred) == 0:
        return 0.0

    # Sort predictions by score
    indices = np.argsort(scores)[::-1]
    boxes_pred = np.array(boxes_pred)[indices]

    map_total = 0

    for iou_thresh in thresholds:
        tp = np.zeros(len(boxes_pred))
        fp = np.zeros(len(boxes_pred))
        gt_matched = np.zeros(len(boxes_true))

        for i, box_p in enumerate(boxes_pred):
            best_iou = 0
            best_gt_idx = -1

            for j, box_t in enumerate(boxes_true):
                iou = calculate_iou(box_p, box_t)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_thresh and gt_matched[best_gt_idx] == 0:
                tp[i] = 1
                gt_matched[best_gt_idx] = 1
            else:
                fp[i] = 1

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (len(boxes_true) + 1e-6)

        # Compute Average Precision (Area under PR curve)
        # Append sentinels
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))

        # Smooth precision
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Integrate
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        map_total += ap

    return map_total / len(thresholds)

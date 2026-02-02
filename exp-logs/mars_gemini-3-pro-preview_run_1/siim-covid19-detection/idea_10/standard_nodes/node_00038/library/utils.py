import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask2bbox(mask, threshold=0.5):
    """
    Converts a probability mask or binary mask into bounding boxes.

    Args:
        mask (np.ndarray or torch.Tensor): The input mask (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: A list of bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Convert xywh to xmin, ymin, xmax, ymax
        boxes.append([x, y, x + w, y + h])

    return boxes


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Boxes are [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        # Handle case where both boxes are points (0 area) but identical
        if inter_area == 0 and box1_area == 0 and box2_area == 0:
            if box1 == box2:
                return 1.0
        return 0.0

    return inter_area / union_area


def calculate_ap_single_class(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculates AP for a single class using VOC 2010 interpolation.

    Args:
        pred_boxes (list): List of numpy arrays for each image.
                           Each array is (N, 5) -> [x1, y1, x2, y2, score]
        true_boxes (list): List of numpy arrays for each image.
                           Each array is (M, 4) -> [x1, y1, x2, y2]
    """
    # Flatten all predictions into a single list of (score, img_idx, box)
    all_preds = []
    for img_idx, boxes in enumerate(pred_boxes):
        if boxes.shape[0] > 0:
            for i in range(boxes.shape[0]):
                # boxes[i] is [x1, y1, x2, y2, score]
                all_preds.append((boxes[i, 4], img_idx, boxes[i, :4]))

    # Sort predictions by confidence score (descending)
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Initialize arrays for TP and FP
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track matched GT boxes
    true_boxes_matched = {}
    total_positives = 0

    for img_idx, boxes in enumerate(true_boxes):
        true_boxes_matched[img_idx] = np.zeros(boxes.shape[0], dtype=bool)
        total_positives += boxes.shape[0]

    if total_positives == 0:
        return 0.0

    # Match predictions to ground truth
    for i, (score, img_idx, pred_box) in enumerate(all_preds):
        gt_boxes = true_boxes[img_idx]
        best_iou = -1.0
        best_gt_idx = -1

        # Find best matching GT
        for gt_idx in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[gt_idx]
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            if not true_boxes_matched[img_idx][best_gt_idx]:
                tp[i] = 1
                true_boxes_matched[img_idx][best_gt_idx] = True
            else:
                fp[i] = 1  # Duplicate
        else:
            fp[i] = 1

    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute Precision and Recall
    recalls = tp_cumsum / total_positives
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # VOC 2010 Interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)


def calculate_map(preds, targets, num_classes=5, iou_threshold=0.5):
    """
    Calculates mAP across multiple classes.

    Args:
        preds (list of dict): Each dict contains 'boxes' (N,4), 'scores' (N,), 'labels' (N,)
        targets (list of dict): Each dict contains 'boxes' (M,4), 'labels' (M,)
        num_classes (int): Number of classes to evaluate.
        iou_threshold (float): IoU threshold.

    Returns:
        float: Mean Average Precision across all classes.
    """
    aps = []

    for c in range(num_classes):
        # Filter predictions and targets for class c
        class_preds = []
        for p in preds:
            mask = p["labels"] == c
            if np.any(mask):
                # Stack boxes and scores: (N, 5)
                boxes = p["boxes"][mask]
                scores = p["scores"][mask]
                class_preds.append(np.column_stack((boxes, scores)))
            else:
                class_preds.append(np.empty((0, 5)))

        class_targets = []
        for t in targets:
            mask = t["labels"] == c
            if np.any(mask):
                class_targets.append(t["boxes"][mask])
            else:
                class_targets.append(np.empty((0, 4)))

        ap = calculate_ap_single_class(class_preds, class_targets, iou_threshold)
        aps.append(ap)

    return np.mean(aps)

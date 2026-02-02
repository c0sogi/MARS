import numpy as np
import cv2
import torch
from library.config import seed_everything


def iou_score(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: List or array [x1, y1, x2, y2]
        box2: List or array [x1, y1, x2, y2]

    Returns:
        float: IoU score between 0.0 and 1.0
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


def get_box_from_mask(mask, threshold=0.5):
    """
    Extracts bounding boxes from a probability mask using contour detection.

    Args:
        mask: Probability mask of shape (H, W) or (1, H, W). Can be numpy array or torch.Tensor.
        threshold: Float threshold to binarize the mask.

    Returns:
        list: List of bounding boxes in format [x_min, y_min, x_max, y_max].
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask.ndim == 3:
        mask = mask[0]

    # Binarize
    mask_bin = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Convert xywh to xmin, ymin, xmax, ymax
        boxes.append([x, y, x + w, y + h])

    return boxes


def calculate_map(
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5
):
    """
    Calculates the mean Average Precision (mAP) at a specific IoU threshold
    using the PASCAL VOC 2010 methodology (all-point interpolation).

    Args:
        pred_boxes: List of lists of boxes (one list per image). Each box is [xmin, ymin, xmax, ymax].
        pred_scores: List of lists of confidence scores.
        pred_labels: List of lists of labels (int or str).
        gt_boxes: List of lists of ground truth boxes.
        gt_labels: List of lists of ground truth labels.
        iou_threshold: IoU threshold to consider a detection a True Positive.

    Returns:
        float: The mAP score.
    """
    # Identify all unique classes present in the ground truth
    unique_classes = set()
    for labels in gt_labels:
        unique_classes.update(labels)

    if not unique_classes:
        return 0.0

    aps = []

    for cls in unique_classes:
        # --- 1. Prepare Ground Truths for this class ---
        # class_gts maps image_index -> {'boxes': array, 'visited': array}
        class_gts = {}
        n_pos = 0

        for i, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            # Filter boxes for current class
            cls_boxes = [b for b, l in zip(boxes, labels) if l == cls]
            class_gts[i] = {
                "boxes": np.array(cls_boxes),
                "visited": np.zeros(len(cls_boxes), dtype=bool),
            }
            n_pos += len(cls_boxes)

        # --- 2. Prepare Predictions for this class ---
        # List of (score, box, img_idx)
        class_preds = []
        for i, (boxes, scores, labels) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            for b, s, l in zip(boxes, scores, labels):
                if l == cls:
                    class_preds.append((float(s), b, i))

        # Sort predictions by confidence score (descending)
        class_preds.sort(key=lambda x: x[0], reverse=True)

        # --- 3. Calculate TP and FP ---
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, (score, box, img_idx) in enumerate(class_preds):
            gt_data = class_gts[img_idx]
            gt_boxes_img = gt_data["boxes"]

            best_iou = -1.0
            best_idx = -1

            if len(gt_boxes_img) > 0:
                # Find best matching GT box
                for j, gt_box in enumerate(gt_boxes_img):
                    iou = iou_score(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

            if best_iou >= iou_threshold:
                if not gt_data["visited"][best_idx]:
                    tp[i] = 1.0
                    gt_data["visited"][best_idx] = True
                else:
                    fp[i] = 1.0  # Duplicate detection
            else:
                fp[i] = 1.0  # False positive (IoU too low or no GT)

        # --- 4. Compute Precision and Recall ---
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        if n_pos > 0:
            recalls = tp_cumsum / n_pos
        else:
            recalls = np.zeros_like(tp_cumsum)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # --- 5. Compute AP (All-point interpolation - VOC 2010) ---
        # Append sentinel values to integrate correctly
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope (make it monotonically decreasing)
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = np.maximum(mpre[i], mpre[i + 1])

        # Integrate area under curve
        # Find points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum of rectangular areas
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        aps.append(ap)

    return np.mean(aps)


def format_prediction_string(labels, scores, boxes):
    """
    Formats predictions into the competition submission string format.

    Args:
        labels: List of label strings or class IDs.
        scores: List of confidence scores.
        boxes: List of boxes [xmin, ymin, xmax, ymax].

    Returns:
        str: Prediction string e.g. "opacity 0.5 100 100 200 200 ..." or "none 1 0 0 1 1"
    """
    if len(labels) == 0:
        return "none 1 0 0 1 1"

    parts = []
    for label, score, box in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = box
        parts.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")

    return " ".join(parts)

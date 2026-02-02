import os
import random
import numpy as np
import torch
import cv2
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_box_from_mask(mask, threshold=0.5):
    """
    Extracts bounding boxes from a probability mask using contour detection.

    Args:
        mask (np.array or torch.Tensor): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Binarize mask
    mask_bin = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # x, y is top-left; w, h are width, height
        # Convert to x_min, y_min, x_max, y_max
        boxes.append([x, y, x + w, y + h])

    return boxes


def iou_bbox(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area < 1e-6:
        return 0.0

    return intersection_area / union_area


def parse_prediction_string(pred_str):
    """
    Parses a prediction string into a list of dictionaries.
    Format: "label confidence x1 y1 x2 y2 ..."
    """
    if pd.isna(pred_str) or pred_str == "":
        return []

    parts = pred_str.split()
    results = []

    # Each prediction consists of 6 tokens: label, conf, x1, y1, x2, y2
    for i in range(0, len(parts), 6):
        if i + 6 > len(parts):
            break

        label = parts[i]
        try:
            conf = float(parts[i + 1])
            x1 = float(parts[i + 2])
            y1 = float(parts[i + 3])
            x2 = float(parts[i + 4])
            y2 = float(parts[i + 5])

            results.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})
        except ValueError:
            continue

    return results


def calculate_ap(tp, fp, n_positives):
    """
    Calculates Average Precision (AP) using the PASCAL VOC 2010 method (Interpolated AP).
    """
    if n_positives == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / n_positives
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Append start and end values for interpolation
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate area under the curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def map_calculation(gt_df, pred_df, iou_threshold=0.5):
    """
    Calculates mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        gt_df (pd.DataFrame): Dataframe with columns ['id', 'PredictionString'].
                              PredictionString format: "label 1 x1 y1 x2 y2 ..."
        pred_df (pd.DataFrame): Dataframe with columns ['id', 'PredictionString'].
                                PredictionString format: "label conf x1 y1 x2 y2 ..."
        iou_threshold (float): IoU threshold for matching predictions to ground truth.

    Returns:
        float: The mAP score.
    """
    # Create dictionaries for O(1) access
    gt_dict = dict(zip(gt_df["id"], gt_df["PredictionString"]))
    pred_dict = dict(zip(pred_df["id"], pred_df["PredictionString"]))

    all_ids = set(gt_dict.keys()) | set(pred_dict.keys())

    # Containers for per-class data
    class_preds = {}  # label -> list of (conf, bbox, image_id)
    class_gts = {}  # label -> dict of image_id -> list of bboxes

    # 1. Parse Ground Truths
    for img_id in all_ids:
        gt_str = gt_dict.get(img_id, "")
        parsed_gt = parse_prediction_string(gt_str)

        for item in parsed_gt:
            label = item["label"]
            bbox = item["bbox"]

            if label not in class_gts:
                class_gts[label] = {}
            if img_id not in class_gts[label]:
                class_gts[label][img_id] = []
            class_gts[label][img_id].append(bbox)

    # 2. Parse Predictions
    for img_id in all_ids:
        pred_str = pred_dict.get(img_id, "")
        parsed_pred = parse_prediction_string(pred_str)

        for item in parsed_pred:
            label = item["label"]
            conf = item["conf"]
            bbox = item["bbox"]

            if label not in class_preds:
                class_preds[label] = []
            class_preds[label].append((conf, bbox, img_id))

    # 3. Calculate AP per class
    aps = []
    all_classes = set(class_gts.keys()) | set(class_preds.keys())

    for label in all_classes:
        # Ground truths for this class
        gts = class_gts.get(label, {})
        n_positives = sum(len(boxes) for boxes in gts.values())

        # Predictions for this class
        preds = class_preds.get(label, [])
        # Sort by confidence descending
        preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # Track matched GT boxes to prevent double counting
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in gts.items()}

        for i, (conf, pred_bbox, img_id) in enumerate(preds):
            if img_id not in gts:
                fp[i] = 1
                continue

            gt_boxes = gts[img_id]
            best_iou = -1
            best_idx = -1

            # Find best matching GT box
            for j, gt_bbox in enumerate(gt_boxes):
                iou = iou_bbox(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            # Check threshold and availability
            if best_iou >= iou_threshold:
                if not gt_matched[img_id][best_idx]:
                    tp[i] = 1
                    gt_matched[img_id][best_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection for same GT
            else:
                fp[i] = 1  # IoU too low

        # Calculate AP for this class
        ap = calculate_ap(tp, fp, n_positives)
        aps.append(ap)

    # 4. Compute mAP
    if not aps:
        return 0.0

    return np.mean(aps)

import os
import random
import numpy as np
import torch
import pandas as pd
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Weighted Boxes Fusion (WBF) Implementation
# -----------------------------------------------------------------------------


def bb_intersection_over_union(A, B):
    """
    Calculate IoU between two boxes A and B [x1, y1, x2, y2].
    """
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def apply_wbf(
    boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5, skip_box_thr=0.0
):
    """
    Applies Weighted Boxes Fusion to ensemble boxes from multiple models or TTA.

    Args:
        boxes_list (list): List of lists of boxes (e.g. [[box1, box2], [box3]]).
        scores_list (list): List of lists of scores.
        labels_list (list): List of lists of labels.
        weights (list, optional): List of weights for each model. Defaults to equal weights.
        iou_thr (float): IoU threshold for clustering boxes.
        skip_box_thr (float): Threshold to exclude low-confidence boxes before fusion.

    Returns:
        tuple: (fused_boxes, fused_scores, fused_labels)
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    if len(boxes_list) == 0:
        return [], [], []

    # 1. Flatten all predictions into a single list
    global_boxes = []
    global_scores = []
    global_labels = []
    global_weights = []

    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            score = scores_list[i][j]
            if score < skip_box_thr:
                continue
            global_boxes.append(boxes_list[i][j])
            global_scores.append(score)
            global_labels.append(labels_list[i][j])
            global_weights.append(weights[i])

    if not global_boxes:
        return [], [], []

    # 2. Sort by score descending
    order = np.argsort(global_scores)[::-1]

    # 3. Cluster boxes
    # Each cluster is a dict with accumulated properties
    clusters = []

    for idx in order:
        box = global_boxes[idx]
        score = global_scores[idx]
        label = global_labels[idx]
        weight = global_weights[idx]

        best_iou = iou_thr
        best_cluster_idx = -1

        # Find matching cluster
        for c_idx, cluster in enumerate(clusters):
            if cluster["label"] != label:
                continue

            # Match against the current weighted average box of the cluster
            iou = bb_intersection_over_union(cluster["avg_box"], box)
            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = c_idx

        if best_cluster_idx != -1:
            # Add to existing cluster
            c = clusters[best_cluster_idx]
            c["boxes"].append(box)
            c["scores"].append(score)
            c["weights"].append(weight)

            # Update average box (weighted by score)
            # Recomputing full average for precision
            w_coords = np.zeros(4)
            score_sum = 0
            for b, s in zip(c["boxes"], c["scores"]):
                w_coords += np.array(b) * s
                score_sum += s
            c["avg_box"] = (w_coords / score_sum).tolist()

        else:
            # Create new cluster
            clusters.append(
                {
                    "boxes": [box],
                    "scores": [score],
                    "weights": [weight],
                    "label": label,
                    "avg_box": box,
                }
            )

    # 4. Compute final fused boxes
    final_boxes = []
    final_scores = []
    final_labels = []

    total_model_weight = sum(weights)

    for c in clusters:
        # Coordinate: Weighted average by score
        # Note: In WBF, coordinates are weighted by score
        w_coords = np.zeros(4)
        score_sum = 0
        for b, s in zip(c["boxes"], c["scores"]):
            w_coords += np.array(b) * s
            score_sum += s
        avg_box = w_coords / score_sum

        # Score: Sum of scores / Sum of all model weights (penalize missing detections)
        # Standard WBF: score = sum(scores) / N_models
        # Weighted WBF: score = sum(scores * weight) / sum(all_weights)
        # Here we treat 'scores' as raw confidence.
        # Since we flattened, we iterate through the items in the cluster.
        # We need to sum the scores (potentially weighted) and divide by total weight.
        # Assuming standard implementation where we just average the scores found and scale by presence?
        # Standard WBF formula: res_score = (sum(scores) / N) -- effectively treating missing as 0.

        # Using simple average over total possible weight to penalize missing boxes
        weighted_score_sum = sum(
            s * 1.0 for s in c["scores"]
        )  # Assuming score already reflects confidence
        # If we want to use model weights for the score importance:
        # weighted_score_sum = sum(s * w for s, w in zip(c['scores'], c['weights']))
        # Let's stick to the standard logic: sum(scores) / N_models

        final_score = sum(c["scores"]) / len(boxes_list)

        final_boxes.append(avg_box.tolist())
        final_scores.append(final_score)
        final_labels.append(c["label"])

    return final_boxes, final_scores, final_labels


# -----------------------------------------------------------------------------
# Metric Calculation (mAP)
# -----------------------------------------------------------------------------


def calculate_ap(rec, prec):
    """
    Compute VOC AP given precision and recall using all-point interpolation.
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve
    # Look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5
):
    """
    Calculate Mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        pred_boxes (list): List of lists of predicted boxes for each image.
        pred_scores (list): List of lists of predicted scores.
        pred_labels (list): List of lists of predicted labels.
        gt_boxes (list): List of lists of ground truth boxes.
        gt_labels (list): List of lists of ground truth labels.
        iou_threshold (float): IoU threshold for a positive match.

    Returns:
        float: The mAP score.
    """
    unique_classes = set()
    for labels in gt_labels:
        unique_classes.update(labels)
    for labels in pred_labels:
        unique_classes.update(labels)

    aps = []

    for cls in unique_classes:
        class_preds = []  # (score, box, image_idx)
        class_gts = {}  # image_idx -> [boxes]
        n_pos = 0

        # Organize GT and Preds by class
        for i in range(len(gt_labels)):
            # GT
            gts = []
            for j, label in enumerate(gt_labels[i]):
                if label == cls:
                    gts.append(gt_boxes[i][j])
            class_gts[i] = np.array(gts)
            n_pos += len(gts)

            # Preds
            for j, label in enumerate(pred_labels[i]):
                if label == cls:
                    class_preds.append((pred_scores[i][j], pred_boxes[i][j], i))

        if n_pos == 0:
            continue

        # Sort predictions by score descending
        class_preds.sort(key=lambda x: x[0], reverse=True)

        TP = np.zeros(len(class_preds))
        FP = np.zeros(len(class_preds))

        # Track detected GTs to avoid double counting
        detected_gt = {i: np.zeros(len(class_gts[i]), dtype=bool) for i in class_gts}

        for d, (score, box_pred, img_idx) in enumerate(class_preds):
            gts = class_gts[img_idx]
            if len(gts) == 0:
                FP[d] = 1
                continue

            # Calculate IoU with all GTs in the image
            ixmin = np.maximum(gts[:, 0], box_pred[0])
            iymin = np.maximum(gts[:, 1], box_pred[1])
            ixmax = np.minimum(gts[:, 2], box_pred[2])
            iymax = np.minimum(gts[:, 3], box_pred[3])

            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin, 0.0)
            inters = iw * ih

            uni = (
                (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
                + (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
                - inters
            )

            ious = inters / uni

            if len(ious) > 0:
                iou_max = np.max(ious)
                j_max = np.argmax(ious)

                if iou_max > iou_threshold:
                    if not detected_gt[img_idx][j_max]:
                        TP[d] = 1
                        detected_gt[img_idx][j_max] = True
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        # Compute Precision and Recall
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        rec = acc_TP / n_pos
        prec = acc_TP / (acc_TP + acc_FP + 1e-6)

        ap = calculate_ap(rec, prec)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return np.mean(aps)


# -----------------------------------------------------------------------------
# String Formatting
# -----------------------------------------------------------------------------


def format_prediction_string(labels, boxes, scores):
    """
    Converts lists of labels, boxes, and scores into the submission format string.
    """
    if len(boxes) == 0:
        return "none 1 0 0 1 1"

    pred_strings = []
    for label, box, score in zip(labels, boxes, scores):
        # Handle label conversion
        l_str = str(label)
        if l_str == "1":
            l_str = "opacity"

        # Skip background if present
        if l_str == "0":
            continue

        # Format: label score xmin ymin xmax ymax
        b = [int(x) for x in box]
        pred_strings.append(f"{l_str} {score:.6f} {b[0]} {b[1]} {b[2]} {b[3]}")

    if len(pred_strings) == 0:
        return "none 1 0 0 1 1"

    return " ".join(pred_strings)


def parse_prediction_string(pred_str):
    """
    Parses a prediction string back into lists of labels, boxes, and scores.
    """
    if pd.isna(pred_str) or pred_str == "none 1 0 0 1 1":
        return [], [], []

    parts = pred_str.split()
    labels = []
    scores = []
    boxes = []

    # Format: label score xmin ymin xmax ymax
    for i in range(0, len(parts), 6):
        labels.append(parts[i])
        scores.append(float(parts[i + 1]))
        boxes.append(
            [
                float(parts[i + 2]),
                float(parts[i + 3]),
                float(parts[i + 4]),
                float(parts[i + 5]),
            ]
        )

    return labels, boxes, scores

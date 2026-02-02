import numpy as np
from library.config import Config, seed_everything


def bb_intersection_over_union(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0
):
    """
    Weighted Boxes Fusion (WBF) implementation.
    Merges overlapping boxes from multiple predictions (e.g., TTA) by averaging coordinates
    weighted by confidence scores.

    Args:
        boxes_list (list): List of lists of boxes [x1, y1, x2, y2] from each model/TTA.
        scores_list (list): List of lists of scores from each model/TTA.
        labels_list (list): List of lists of labels from each model/TTA.
        weights (list, optional): Weights for each model. Defaults to None (equal weights).
        iou_thr (float): IoU threshold for clustering boxes.
        skip_box_thr (float): Threshold to exclude boxes with low confidence before fusion.

    Returns:
        tuple: (boxes, scores, labels) as numpy arrays.
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    box_model_indices = []

    # 1. Filter and flatten predictions from all models/TTA steps
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            score = scores_list[i][j]
            if score < skip_box_thr:
                continue
            filtered_boxes.append(boxes_list[i][j])
            filtered_scores.append(score)
            filtered_labels.append(labels_list[i][j])
            box_model_indices.append(i)

    if len(filtered_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    filtered_boxes = np.array(filtered_boxes)
    filtered_scores = np.array(filtered_scores)
    filtered_labels = np.array(filtered_labels)
    box_model_indices = np.array(box_model_indices)

    # 2. Sort all boxes globally by score (descending)
    order = filtered_scores.argsort()[::-1]
    filtered_boxes = filtered_boxes[order]
    filtered_scores = filtered_scores[order]
    filtered_labels = filtered_labels[order]
    box_model_indices = box_model_indices[order]

    clusters = []

    # 3. Cluster boxes
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        score = filtered_scores[i]
        label = filtered_labels[i]
        model_idx = box_model_indices[i]

        matching_cluster_idx = -1
        best_iou = -1

        # Find best matching cluster
        for c_idx, cluster in enumerate(clusters):
            if cluster["label"] != label:
                continue

            # Match against the current weighted average box of the cluster
            iou = bb_intersection_over_union(box, cluster["avg_box"])
            if iou > iou_thr:
                if iou > best_iou:
                    best_iou = iou
                    matching_cluster_idx = c_idx

        if matching_cluster_idx != -1:
            # Add to existing cluster
            c = clusters[matching_cluster_idx]
            w = weights[model_idx]

            c["boxes"].append(box)
            c["scores"].append(score)
            c["model_indices"].append(model_idx)

            # Update weighted average box
            c["sum_score_weighted"] += score * w
            c["weighted_sum_coords"] += box * score * w
            c["avg_box"] = c["weighted_sum_coords"] / c["sum_score_weighted"]
        else:
            # Create new cluster
            w = weights[model_idx]
            clusters.append(
                {
                    "label": label,
                    "boxes": [box],
                    "scores": [score],
                    "model_indices": [model_idx],
                    "sum_score_weighted": score * w,
                    "weighted_sum_coords": box * score * w,
                    "avg_box": box,
                }
            )

    # 4. Compute final results
    res_boxes = []
    res_scores = []
    res_labels = []

    total_weight = np.sum(weights)

    for c in clusters:
        # Calculate final score: sum of (best score per model * weight) / total_weight
        # This penalizes clusters that are not supported by all models
        model_best_scores = {}
        for s, m_idx in zip(c["scores"], c["model_indices"]):
            if m_idx not in model_best_scores:
                model_best_scores[m_idx] = s
            else:
                model_best_scores[m_idx] = max(model_best_scores[m_idx], s)

        final_score_numerator = 0
        for m_idx, s in model_best_scores.items():
            final_score_numerator += s * weights[m_idx]

        final_score = final_score_numerator / total_weight

        res_boxes.append(c["avg_box"])
        res_scores.append(final_score)
        res_labels.append(c["label"])

    return np.array(res_boxes), np.array(res_scores), np.array(res_labels)


def format_prediction_string(labels, boxes, scores):
    """
    Format predictions into the competition submission string format.

    Args:
        labels (list/array): Class indices.
        boxes (list/array): Bounding boxes [x1, y1, x2, y2].
        scores (list/array): Confidence scores.

    Returns:
        str: Prediction string (e.g., "opacity 0.5 100 100 200 200 ...").
             Returns "none 1 0 0 1 1" if no boxes are provided.
    """
    if len(boxes) == 0:
        return "none 1 0 0 1 1"

    pred_strings = []
    for label, box, score in zip(labels, boxes, scores):
        # Map label index to class name
        # Assuming label 1 corresponds to index 0 in DETECTION_LABELS ("opacity")
        # If label is 0 (and DETECTION_LABELS has content), we map it to index 0 as fallback
        if label > 0 and (label - 1) < len(Config.DETECTION_LABELS):
            class_name = Config.DETECTION_LABELS[int(label) - 1]
        elif label == 0 and len(Config.DETECTION_LABELS) > 0:
            class_name = Config.DETECTION_LABELS[0]
        else:
            class_name = "opacity"

        x1, y1, x2, y2 = box
        pred_strings.append(
            f"{class_name} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
        )

    return " ".join(pred_strings)

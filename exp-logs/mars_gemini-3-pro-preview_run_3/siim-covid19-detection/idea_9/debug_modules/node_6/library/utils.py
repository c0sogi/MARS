import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
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


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list/array): [x1, y1, x2, y2]
        box2 (list/array): [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0.0 and 1.0
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


def weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0
):
    """
    Implements Weighted Boxes Fusion (WBF) to merge predictions from multiple sources (e.g., TTA).

    Args:
        boxes_list (list): List of lists of boxes [x1, y1, x2, y2]. Dimension: [Models, Boxes, 4]
        scores_list (list): List of lists of scores. Dimension: [Models, Boxes]
        labels_list (list): List of lists of labels. Dimension: [Models, Boxes]
        weights (list, optional): Weights for each model/source. Defaults to equal weights.
        iou_thr (float): IoU threshold for clustering boxes.
        skip_box_thr (float): Confidence threshold to skip low-confidence boxes before fusion.

    Returns:
        tuple: (fused_boxes, fused_scores, fused_labels) as numpy arrays.
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    if len(weights) != len(boxes_list):
        raise ValueError("Length of weights must match length of boxes_list")

    # Flatten all predictions into a single list
    all_boxes = []
    for i in range(len(boxes_list)):
        w = weights[i]
        for j in range(len(boxes_list[i])):
            score = scores_list[i][j]
            if score < skip_box_thr:
                continue
            box = boxes_list[i][j]
            label = labels_list[i][j]
            all_boxes.append({"box": box, "score": score, "label": label, "weight": w})

    # Sort boxes by score in descending order
    all_boxes.sort(key=lambda x: x["score"], reverse=True)

    clusters = []

    # Cluster boxes
    for item in all_boxes:
        box = item["box"]
        score = item["score"]
        label = item["label"]
        weight = item["weight"]

        best_iou = -1
        best_idx = -1

        # Find the best matching cluster
        for k, cluster in enumerate(clusters):
            if cluster["label"] != label:
                continue

            # Calculate IoU with the weighted average box of the cluster
            c_boxes = np.array(cluster["boxes"])
            c_scores = np.array(cluster["scores"])
            avg_box = np.average(c_boxes, axis=0, weights=c_scores)

            iou = calculate_iou(box, avg_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = k

        # Add to cluster or create new one
        if best_iou > iou_thr:
            clusters[best_idx]["boxes"].append(box)
            clusters[best_idx]["scores"].append(score)
            clusters[best_idx]["weights"].append(weight)
        else:
            clusters.append(
                {"boxes": [box], "scores": [score], "weights": [weight], "label": label}
            )

    final_boxes = []
    final_scores = []
    final_labels = []

    overall_weight_sum = sum(weights)

    # Calculate fused boxes and scores
    for cluster in clusters:
        c_boxes = np.array(cluster["boxes"])
        c_scores = np.array(cluster["scores"])

        # Weighted average of coordinates
        avg_box = np.average(c_boxes, axis=0, weights=c_scores)

        # Rescaled score: Sum of scores / Sum of weights
        # This penalizes boxes that are not predicted by all models
        final_score = np.sum(c_scores) / overall_weight_sum

        final_boxes.append(avg_box)
        final_scores.append(final_score)
        final_labels.append(cluster["label"])

    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)


def get_study_prediction_string(label, conf):
    """
    Formats the study-level prediction string.
    Format: class_id confidence 0 0 1 1
    """
    return f"{label} {conf:.6f} 0 0 1 1"


def get_image_prediction_string(boxes, scores):
    """
    Formats the image-level prediction string.
    Format: opacity conf xmin ymin xmax ymax ...
    If no boxes, returns: none 1 0 0 1 1
    """
    if len(boxes) == 0:
        return "none 1 0 0 1 1"

    parts = []
    for box, score in zip(boxes, scores):
        # Ensure box coordinates are valid
        parts.append(
            f"opacity {score:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        )

    return " ".join(parts)


class MAPCalculator:
    """
    Helper class to calculate PASCAL VOC mAP (Average Precision) over a dataset.
    Accumulates predictions and targets, then computes AP.
    """

    def __init__(self):
        self.predictions = []  # List of tuples (score, is_tp)
        self.n_positives = 0

    def update(self, pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
        """
        Updates the calculator with predictions and ground truth for a single image.

        Args:
            pred_boxes (array): Predicted boxes [N, 4]
            pred_scores (array): Predicted scores [N]
            gt_boxes (array): Ground truth boxes [M, 4]
            iou_threshold (float): IoU threshold for a match to be considered a True Positive.
        """
        self.n_positives += len(gt_boxes)

        if len(pred_boxes) == 0:
            return

        # Sort predictions by score descending
        indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[indices]
        pred_scores = pred_scores[indices]

        gt_matched = np.zeros(len(gt_boxes))

        for box, score in zip(pred_boxes, pred_scores):
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for j, g_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = calculate_iou(box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                self.predictions.append((score, 1))  # True Positive
                gt_matched[best_gt_idx] = 1
            else:
                self.predictions.append((score, 0))  # False Positive

    def compute(self):
        """
        Computes the Average Precision (AP) using the accumulated data.
        """
        if self.n_positives == 0:
            return 0.0

        # Sort all predictions by score descending
        self.predictions.sort(key=lambda x: x[0], reverse=True)

        tps = np.array([x[1] for x in self.predictions])
        fps = 1 - tps

        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)

        recalls = tp_cumsum / self.n_positives
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Compute AP using 11-point interpolation / area under curve (VOC style)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

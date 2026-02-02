import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_train_transform():
    """
    Returns the training transformations including photometric augmentations.
    Expects bboxes in 'pascal_voc' format (x_min, y_min, x_max, y_max).
    """
    return A.Compose(
        [
            # Photometric Augmentations
            # Forces model to learn shape-invariant features, robust to paper aging/quality
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            # Normalize pixel values to [0, 1] as expected by standard backbones
            A.ToFloat(max_value=255.0),
            # Convert to PyTorch Tensor (C, H, W)
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def get_valid_transform():
    """
    Returns the validation transformations.
    Expects bboxes in 'pascal_voc' format (x_min, y_min, x_max, y_max).
    """
    return A.Compose(
        [
            # Normalize pixel values to [0, 1]
            A.ToFloat(max_value=255.0),
            # Convert to PyTorch Tensor (C, H, W)
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    Handles variable number of bounding boxes per image by zipping the list.
    """
    return tuple(zip(*batch))


def calculate_f1_score(preds, targets, score_threshold=Config.SCORE_THRESH):
    """
    Calculates the modified F1 score based on center points.

    Metric Definition:
    To score a true positive, the predicted center point coordinates must be
    within the ground truth bounding box and have a matching label.

    Args:
        preds: List of dicts with keys 'boxes', 'labels', 'scores'.
        targets: List of dicts with keys 'boxes', 'labels'.
        score_threshold: Confidence threshold for predictions.

    Returns:
        dict: {'f1': float, 'precision': float, 'recall': float}
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Iterate over the batch
    for pred, target in zip(preds, targets):
        # Move inputs to CPU and numpy for processing
        gt_boxes = target["boxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        pred_boxes = pred["boxes"].detach().cpu().numpy()
        pred_labels = pred["labels"].detach().cpu().numpy()
        pred_scores = pred["scores"].detach().cpu().numpy()

        # Filter predictions by score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]

        # Enforce maximum detections per image limit
        if len(pred_boxes) > Config.DETECTIONS_PER_IMG:
            # Sort by score descending and take top N
            indices = np.argsort(pred_scores)[::-1][: Config.DETECTIONS_PER_IMG]
            pred_boxes = pred_boxes[indices]
            pred_labels = pred_labels[indices]
            pred_scores = pred_scores[indices]

        # Convert predicted boxes to center points (x, y)
        # Boxes are in pascal_voc format: x1, y1, x2, y2
        pred_centers_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        pred_centers_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0

        matched_gt_indices = set()

        # Iterate through each prediction to find a match
        for i in range(len(pred_labels)):
            px = pred_centers_x[i]
            py = pred_centers_y[i]
            plabel = pred_labels[i]

            match_found = False

            # Check against all ground truth boxes
            for j in range(len(gt_labels)):
                # Skip if this GT is already matched
                if j in matched_gt_indices:
                    continue

                glabel = gt_labels[j]
                gx1, gy1, gx2, gy2 = gt_boxes[j]

                # Check label match and spatial inclusion
                if plabel == glabel:
                    if gx1 <= px <= gx2 and gy1 <= py <= gy2:
                        matched_gt_indices.add(j)
                        match_found = True
                        break

            if match_found:
                tp_total += 1
            else:
                fp_total += 1

        # False Negatives are the number of unmatched ground truths
        fn_total += len(gt_labels) - len(matched_gt_indices)

    # Calculate metrics with zero-division safety
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"f1": f1, "precision": precision, "recall": recall}

import numpy as np
import torch
from library.utils import compute_salt_metric


def calculate_iou(preds, targets, threshold=0.5):
    """
    Computes the mean Intersection over Union (IoU) for a batch of predictions.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or logits.
        targets (torch.Tensor or np.ndarray): Ground truth masks.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        float: Mean IoU for the batch.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions and targets
    # Assuming targets are already binary (0 or 1), but enforcing type safety
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    batch_size = preds_bin.shape[0]
    ious = []

    for i in range(batch_size):
        # Flatten arrays to ensure correct calculation regardless of shape (C, H, W) vs (H, W)
        p = preds_bin[i].flatten()
        t = targets_bin[i].flatten()

        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()

        if union == 0:
            # If union is 0, it means both ground truth and prediction are empty.
            # In this segmentation task, correctly predicting "nothing" is a perfect score (1.0).
            ious.append(1.0)
        else:
            ious.append(intersection / union)

    return np.mean(ious)


def calculate_map_at_thresholds(preds, targets, threshold=0.5):
    """
    Computes the Mean Average Precision at IoU thresholds [0.5, ..., 0.95].
    This matches the competition metric definition.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or logits.
        targets (torch.Tensor or np.ndarray): Ground truth masks.
        threshold (float): Threshold for binarizing predictions before metric calculation.
                           This is the pixel-level classification threshold.

    Returns:
        float: Mean Average Precision over the batch.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    batch_size = preds_bin.shape[0]
    scores = []

    for i in range(batch_size):
        # Extract single image masks
        # Handle (C, H, W) where C=1, or (H, W)
        if preds_bin.ndim == 4:
            p = preds_bin[i, 0]
            t = targets_bin[i, 0]
        elif preds_bin.ndim == 3:
            p = preds_bin[i]
            t = targets_bin[i]
        else:
            # Fallback for flat or other shapes
            p = preds_bin[i]
            t = targets_bin[i]

        # Use the utility function which implements the exact competition metric logic
        # (Sweeping thresholds 0.5 to 0.95)
        score = compute_salt_metric(p, t)
        scores.append(score)

    return np.mean(scores)

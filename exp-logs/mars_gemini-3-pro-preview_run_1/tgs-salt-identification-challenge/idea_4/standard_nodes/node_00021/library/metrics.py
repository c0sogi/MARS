import numpy as np
import torch


def calculate_iou_map(y_pred, y_true, threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at different IoU thresholds.

    The metric sweeps over a range of IoU thresholds from 0.5 to 0.95 with a step size of 0.05.
    At each threshold, a precision value is calculated. The average precision of a single image
    is the mean of these precision values. The final metric is the mean over all images.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or binary masks.
                                             Shape: (N, H, W) or (N, H, W, 1).
        y_true (torch.Tensor or np.ndarray): Ground truth binary masks.
                                             Shape: (N, H, W) or (N, H, W, 1).
        threshold (float): Threshold to binarize predicted probabilities. Default: 0.5.

    Returns:
        float: The mean average precision (mAP) score.
    """
    # 1. Convert inputs to numpy arrays if they are tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # 2. Flatten the spatial dimensions to treat masks as vectors
    # This handles (N, H, W) and (N, H, W, 1) shapes automatically
    batch_size = len(y_true)
    y_pred = y_pred.reshape(batch_size, -1)
    y_true = y_true.reshape(batch_size, -1)

    # 3. Binarize predictions and ground truth
    # y_pred is assumed to be probabilities (0-1). If logits, caller must sigmoid first.
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > 0.5).astype(np.uint8)

    # 4. Calculate Intersection and Union per image
    intersection = (y_pred_bin & y_true_bin).sum(axis=1)
    union = (y_pred_bin | y_true_bin).sum(axis=1)

    # 5. Calculate IoU
    # Handle the special case where Union is 0 (both GT and Pred are empty).
    # In this case, the match is perfect, so IoU is defined as 1.0.
    iou = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=float),
        where=union != 0,
    )

    # 6. Calculate Precision at each threshold
    # Thresholds: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    thresholds = np.arange(0.5, 0.96, 0.05)

    # We compare the IoU of each image against all thresholds.
    # iou shape: (Batch_Size,)
    # thresholds shape: (10,)
    # Result shape: (Batch_Size, 10)
    # matches[i, j] is True if iou[i] > thresholds[j]
    matches = iou[:, np.newaxis] > thresholds[np.newaxis, :]

    # 7. Calculate Average Precision per image
    # The score for an image is the mean of the boolean matches (0 or 1) over thresholds.
    image_scores = matches.mean(axis=1)

    # 8. Calculate Mean Average Precision over the batch
    final_score = image_scores.mean()

    return float(final_score)

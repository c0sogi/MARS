import numpy as np
import torch
from library.config import *


def calculate_fbeta(y_pred_bin, y_true, beta=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary masks using the Sørensen-Dice coefficient variant.

    Formula: ((1 + beta^2) * p * r) / (beta^2 * p + r)
    where p = precision, r = recall.

    Args:
        y_pred_bin (np.ndarray or torch.Tensor): Binary predictions (0 or 1).
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels (0 or 1).
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The F-beta score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_pred_bin, torch.Tensor):
        y_pred_bin = y_pred_bin.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Flatten arrays to 1D for metric calculation
    y_pred_bin = y_pred_bin.flatten().astype(np.float32)
    y_true = y_true.flatten().astype(np.float32)

    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum(y_pred_bin * y_true)
    fp = np.sum(y_pred_bin * (1 - y_true))
    fn = np.sum((1 - y_pred_bin) * y_true)

    # Calculate Precision and Recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F-beta
    beta_sq = beta**2
    fbeta = ((1 + beta_sq) * precision * recall) / (
        beta_sq * precision + recall + epsilon
    )

    return float(fbeta)


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.
    The pixels are numbered from left to right, then top to bottom (Row-major).
    Output format is a space-delimited string of 'start length' pairs.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited run-length encoding.
    """
    # Ensure mask is binary and flatten in row-major order
    pixels = mask.flatten()

    # We pad the pixels with 0 at start and end to detect all transitions
    # np.concatenate is efficient for this
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains start indices of segments.
    # Even indices (0, 2, ...) are starts of 1s (because we padded with 0).
    # Odd indices (1, 3, ...) are ends of 1s (starts of 0s).

    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def calibrate_threshold(y_true, y_pred_probs, beta=0.5, step=0.01):
    """
    Finds the optimal probability threshold to maximize the F-beta score on validation data.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred_probs (np.ndarray or torch.Tensor): Predicted probabilities (0.0 to 1.0).
        beta (float): Beta value for F-score optimization.
        step (float): Step size for iterating thresholds.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    # Flatten
    y_true = y_true.flatten()
    y_pred_probs = y_pred_probs.flatten()

    best_threshold = 0.5
    best_score = -1.0

    # Iterate through thresholds
    # We use a range slightly inside [0, 1] to avoid trivial all-0 or all-1 masks if possible
    thresholds = np.arange(0.01, 1.0, step)

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs >= thresh).astype(np.uint8)

        # Calculate score
        score = calculate_fbeta(y_pred_bin, y_true, beta=beta)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Print full precision as requested
    print(
        f"Calibration Complete. Best Threshold: {best_threshold}, Best F{beta}: {best_score}"
    )

    return best_threshold, best_score

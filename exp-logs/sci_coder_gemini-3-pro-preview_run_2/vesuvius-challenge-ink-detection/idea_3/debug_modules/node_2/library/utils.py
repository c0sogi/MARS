import numpy as np
import torch


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom (row-major).

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width), where 1 indicates ink.

    Returns:
        str: Space-delimited list of pairs <start> <length>.
    """
    # Flatten the mask in row-major order
    pixels = mask.flatten()

    # Prepend and append 0 to detect transitions at the start and end of the array
    # This simplifies the logic for finding runs of 1s
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the start indices (0->1 transitions)
    # runs[1::2] are the end indices (1->0 transitions)
    # The RLE format requires length, so we subtract start from end
    runs[1::2] -= runs[0::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, threshold=0.5, beta=0.5, smooth=1e-6):
    """
    Calculates the F-beta score for binary segmentation.

    The F0.5 score weights precision higher than recall.
    Formula: ((1 + beta^2) * p * r) / (beta^2 * p + r)

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities (0 to 1).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels (0 or 1).
        threshold (float): Threshold for binarizing predictions.
        beta (float): Beta value for the F-score (default 0.5).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Convert PyTorch tensors to NumPy arrays for calculation
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions
    preds_bin = (preds > threshold).astype(float)
    targets = targets.astype(float)

    # Calculate True Positives (tp), False Positives (fp), False Negatives (fn)
    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    # Calculate F-beta score
    fbeta = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + smooth)
    )

    return fbeta


def optimize_threshold(preds, targets, beta=0.5, num_steps=100):
    """
    Finds the optimal binarization threshold that maximizes the F-beta score.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities.
        targets (torch.Tensor or np.ndarray): Ground truth labels.
        beta (float): Beta value for the F-score.
        num_steps (int): Number of steps to search between 0.01 and 0.99.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Convert to NumPy once to avoid repeated overhead
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    best_threshold = 0.5
    best_score = 0.0

    # Search range from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for th in thresholds:
        score = fbeta_score(preds, targets, threshold=th, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = th

    return best_threshold, best_score

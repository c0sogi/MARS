import numpy as np
import torch
from library.config import Config, set_seed


def f05_score(preds, targets, threshold=0.5, epsilon=1e-7):
    """
    Computes the F0.5 score for binary segmentation.

    The F0.5 score weights precision higher than recall.
    Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (probabilities or logits).
        targets (torch.Tensor or np.ndarray): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The F0.5 score.
    """
    # Convert to tensor if numpy array
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Move to CPU for calculation if not already
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # Binarize predictions
    y_pred = (preds > threshold).float().view(-1)
    y_true = targets.float().view(-1)

    # Calculate confusion matrix components
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    # Beta = 0.5
    beta = 0.5
    beta_sq = beta**2

    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def rle_encode(mask):
    """
    Run-Length Encode a binary mask.

    The metric checks that pairs are sorted, positive, and decoded pixel values
    are not duplicated. The pixels are numbered from left to right, then top to bottom:
    1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates ink.

    Returns:
        str: Space-delimited run-length encoding string (e.g., '1 3 10 5').
    """
    # Ensure input is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Flatten row-major (C-style) as per "left to right, then top to bottom"
    pixels = mask.flatten()

    # Pad with 0s at start and end to detect all transitions
    # We use 0 as the "no ink" value
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    # pixels[1:] != pixels[:-1] gives boolean array of transitions
    # np.where returns indices in the sliced array
    # We add 1 to adjust for the padding at the start
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs now contains [start_1, end_1, start_2, end_2, ...]
    # The submission format requires [start_1, length_1, start_2, length_2, ...]
    # Length = end - start
    # Note: The indices are already 1-based relative to the original image
    # because of the padding logic.
    # Example: 0 1 1 0. Padded: 0 0 1 1 0 0.
    # Diff at index 1 (0->1) and index 3 (1->0).
    # runs = [2, 4].
    # Original image indices: 0, 1, 2, 3. 1s are at 1, 2.
    # 1-based indices: 2, 3.
    # Start is 2. Length is 4 - 2 = 2. Correct.

    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

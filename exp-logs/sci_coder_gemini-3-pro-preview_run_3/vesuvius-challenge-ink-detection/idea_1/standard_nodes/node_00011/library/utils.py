import numpy as np
import torch
from library.config import seed_everything


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The output is a space-delimited list of pairs (start, length).
    The pixels are numbered from left to right, then top to bottom:
    1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (height, width).

    Returns:
        str: Space-delimited string of RLE pairs.
    """
    # Flatten the mask in row-major order (C-style)
    pixels = mask.flatten()

    # Prepend and append 0 to detect transitions at the start and end of the array
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array.
    # Since we padded with one 0 at the start, the index in the padded array
    # corresponds exactly to the 1-based index of the pixel in the original array.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains indices of transitions.
    # Even indices (0, 2, ...) correspond to starts of ink runs (0->1)
    # Odd indices (1, 3, ...) correspond to ends of ink runs (1->0)

    # Calculate lengths: length = end_pos - start_pos
    # We modify the odd indices in place to store lengths instead of end positions
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score for binary segmentation.

    The F0.5 score weights precision higher than recall.

    Args:
        preds (torch.Tensor): Predicted probabilities of shape (B, ...).
        targets (torch.Tensor): Ground truth binary masks of shape (B, ...).
        beta (float): Weight of precision in harmonic mean. Default is 0.5.
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The computed F-beta score.
    """
    # Binarize predictions based on threshold
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    # Flatten tensors to compute metrics over the entire batch
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    # Calculate True Positives, False Positives, False Negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    beta_sq = beta**2

    # Calculate F-beta score
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()

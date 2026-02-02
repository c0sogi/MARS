import numpy as np
import torch
from library.config import setup_reproducibility


def set_seed(seed):
    """
    Sets the random seed for reproducibility by delegating to the config library.

    Args:
        seed (int): The seed value to use for random, numpy, and torch.
    """
    setup_reproducibility(seed)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score for binary segmentation tasks.

    The F-beta score weights precision beta times as much as recall.
    For this task, beta=0.5 puts more emphasis on precision.

    Args:
        preds (torch.Tensor): Predicted probabilities or values.
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): The beta parameter (default: 0.5).
        threshold (float): Threshold to binarize predictions (default: 0.5).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The computed F-beta score.
    """
    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()

    beta_sq = beta**2

    # Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def rle_encoding(x):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The format is a space-delimited list of pairs <start> <length>.
    Pixels are numbered from left to right, then top to bottom, starting at 1.

    Args:
        x (numpy.ndarray): Binary mask (0 for background, 1 for ink).

    Returns:
        str: Space-delimited string of run-length encoded values.
    """
    # Flatten the 2D mask to 1D (row-major order)
    pixels = x.flatten()

    # Pad with zeros at the beginning and end to detect transitions efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # The +1 adjusts for the padding at the start and 0-based indexing
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths:
    # runs[::2] are start indices (0->1 transitions)
    # runs[1::2] are end indices (1->0 transitions)
    # Length = End - Start
    runs[1::2] -= runs[::2]

    # Join into a space-delimited string
    return " ".join(str(val) for val in runs)

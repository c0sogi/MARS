import numpy as np
import torch
from library.config import setup_reproducibility


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the setup_reproducibility function from the library config.

    Args:
        seed (int): The seed value to use.
    """
    setup_reproducibility(seed)


def fbeta_score(preds, targets, threshold=0.5, beta=0.5, smooth=1e-6):
    """
    Calculates the F-beta score for binary segmentation.

    The F-beta score is a weighted harmonic mean of precision and recall.
    F0.5 (beta=0.5) weights precision higher than recall.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities (0-1).
        targets (torch.Tensor or np.ndarray): Ground truth binary masks (0 or 1).
        threshold (float): Threshold to convert probabilities to binary mask.
        beta (float): The beta value for the F-score (default 0.5).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    # Flatten tensors to compute global score for the batch
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    # Calculate True Positives, False Positives, False Negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    # Calculate F-beta score
    # Formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)

    return score.item()


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    The format is a space-delimited list of pairs: start_index run_length.
    Pixels are numbered from left to right, then top to bottom (1-based indexing).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (Height, Width).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten the mask (row-major order: left-to-right, top-to-bottom)
    pixels = mask.flatten()

    # Pad with zeros at start and end to detect all transitions
    # We use integers to ensure safe comparison
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains start indices of 1s and start indices of 0s (alternating)
    # Even indices (0, 2, ...) in 'runs' are starts of 1s (ink)
    # Odd indices (1, 3, ...) in 'runs' are starts of 0s (end of ink)

    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    # Join into a space-delimited string
    return " ".join(str(x) for x in runs)

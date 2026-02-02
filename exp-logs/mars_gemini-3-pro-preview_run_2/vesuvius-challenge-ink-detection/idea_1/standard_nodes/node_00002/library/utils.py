import numpy as np
import torch
from library.config import Config, set_seed


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the set_seed function from library.config.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score for binary segmentation.

    The F-beta score weights precision higher than recall when beta < 1.
    Formula: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    Stable Formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or logits.
        targets (torch.Tensor or np.ndarray): Ground truth binary masks.
        beta (float): The beta parameter for F-score (default 0.5).
        threshold (float): Threshold to binarize predictions (default 0.5).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Convert inputs to torch.Tensor if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Binarize predictions based on the threshold
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    # Flatten tensors to compute global TP, FP, FN
    preds_flat = preds_bin.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    # Calculate F-beta score using the numerically stable formula
    numerator = (1 + beta**2) * tp
    denominator = (1 + beta**2) * tp + (beta**2 * fn) + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def rle_encode(img):
    """
    Run-Length Encode (RLE) a binary mask.

    Converts a binary mask into a space-delimited string of pairs (start, length).
    The pixels are numbered from left to right, then top to bottom (Row-Major), 1-indexed.

    Args:
        img (np.ndarray): Binary mask (0s and 1s) of shape (height, width).

    Returns:
        str: Space-delimited run-length encoding string.
    """
    # Flatten the image in row-major order (C-style)
    pixels = img.flatten()

    # Prepend and append 0 to detect runs at the start and end efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # The +1 adjusts for the prepended 0 and converts to 1-based indexing
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the start indices
    # runs[1::2] are the end indices (exclusive in 0-based logic, but here they represent the next change)
    # The length of the run is (end_index - start_index)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

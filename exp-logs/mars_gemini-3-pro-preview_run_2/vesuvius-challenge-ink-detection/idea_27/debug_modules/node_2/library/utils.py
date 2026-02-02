import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def fbeta_score(predictions, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score, which is the weighted harmonic mean of precision and recall.
    The F0.5 score weights precision higher than recall.

    Args:
        predictions (torch.Tensor): Predicted probabilities (0 to 1).
        targets (torch.Tensor): Ground truth binary masks (0 or 1).
        beta (float): The beta value for the F-score. Defaults to 0.5.
        threshold (float): Threshold to binarize predictions. Defaults to 0.5.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The computed F-beta score.
    """
    # Binarize predictions
    preds_bin = (predictions > threshold).float()
    targets_bin = targets.float()

    # Flatten tensors to calculate global metrics (or per-image if needed, but usually global for batch)
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)
    return score.item()


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The output format is a space-delimited list of pairs (start, length).
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1).

    Args:
        img (numpy.ndarray): Binary mask image (0s and 1s).

    Returns:
        str: Run-length encoded string.
    """
    # Flatten the image row-major (standard numpy flatten)
    pixels = img.flatten()

    # We prepend and append 0 to detect runs that start at the beginning or end at the end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The indices in 'runs' alternate between start of a run (0->1) and end of a run (1->0)
    # Start indices are at even positions (0, 2, 4...)
    # End indices are at odd positions (1, 3, 5...)
    # Length is End - Start
    runs[1::2] -= runs[::2]

    # Convert to string space-delimited
    return " ".join(str(x) for x in runs)

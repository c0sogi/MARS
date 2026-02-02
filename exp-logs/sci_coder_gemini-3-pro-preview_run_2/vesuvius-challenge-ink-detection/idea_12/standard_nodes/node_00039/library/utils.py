import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Ensures deterministic behavior for the stabilized optimization protocol.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms are preferred for stability, though may impact performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fbeta_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculates the F-beta score, which is the competition metric (F0.5).
    F-beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)

    The F0.5 score weights precision higher than recall.

    Args:
        preds: Tensor of probabilities or logits.
        targets: Tensor of ground truth (0 or 1).
        beta: Weight of precision in harmonic mean (default 0.5).
        threshold: Threshold for binarizing predictions.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Scalar tensor containing the F-beta score.
    """
    # Flatten predictions and targets to compute global metric
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Binarize predictions based on threshold
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)
    return score


def dice_coef(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculates the Dice Coefficient (F1 Score) for monitoring.
    Dice = 2 * |A intersect B| / (|A| + |B|)

    Args:
        preds: Tensor of probabilities.
        targets: Tensor of ground truth.
        threshold: Threshold for binarizing predictions.
        smooth: Smoothing factor.

    Returns:
        Scalar tensor containing the Dice coefficient.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    preds_bin = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum()

    score = (2.0 * intersection + smooth) / (union + smooth)
    return score


def rle_encoding(mask: np.ndarray) -> str:
    """
    Converts a binary mask to Run-Length Encoding (RLE) format required for submission.
    The pixels are numbered from left to right, then top to bottom (Row-Major).

    Args:
        mask: Binary mask (2D numpy array), where 1 indicates ink and 0 indicates background.

    Returns:
        Space-delimited string of pairs: 'start length start length ...'
    """
    # Flatten the mask (row-major order / C-style)
    pixels = mask.flatten()

    # Add sentinel values (0) at the beginning and end to detect all transitions
    # We want to find runs of 1s.
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start index of the first run of 1s
    # runs[1] is the start index of the subsequent run of 0s (which marks the end of the 1s)
    # The length of the run is runs[1] - runs[0]
    # We update the even indices (ends) to store lengths instead of end positions
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

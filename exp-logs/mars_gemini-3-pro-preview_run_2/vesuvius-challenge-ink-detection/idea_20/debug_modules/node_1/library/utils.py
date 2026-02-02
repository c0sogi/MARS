import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, Numpy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encoding(x):
    """
    Converts a binary mask into Run-Length Encoding (RLE) string format.
    The pixels are numbered from left to right, then top to bottom (row-major).

    Args:
        x (numpy.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Ensure input is a numpy array
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    pixels = x.flatten()
    # Add 0s at start and end to detect transitions efficiently
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coef(
    y_pred, y_true, threshold: float = Config.THRESHOLD, smooth: float = 1e-6
):
    """
    Calculates the Sørensen-Dice coefficient.

    Args:
        y_pred (torch.Tensor): Predicted probabilities.
        y_true (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to convert probabilities to binary.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Apply threshold to get binary predictions
    y_pred_bin = (y_pred > threshold).float()
    y_true_f = y_true.float()

    # Flatten tensors
    y_pred_f = y_pred_bin.view(-1)
    y_true_f = y_true_f.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def fbeta_score(
    y_pred,
    y_true,
    beta: float = 0.5,
    threshold: float = Config.THRESHOLD,
    smooth: float = 1e-6,
):
    """
    Calculates the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    Args:
        y_pred (torch.Tensor): Predicted probabilities.
        y_true (torch.Tensor): Ground truth labels.
        beta (float): The beta parameter (weight of precision vs recall).
                      beta < 1 lends more weight to precision.
        threshold (float): Threshold to convert probabilities to binary.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Apply threshold to get binary predictions
    y_pred_bin = (y_pred > threshold).float()
    y_true_f = y_true.float()

    # Flatten tensors
    y_pred_f = y_pred_bin.view(-1)
    y_true_f = y_true_f.view(-1)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (y_pred_f * y_true_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()

    # Calculate F-beta
    # Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)
    return score.item()

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        img (np.ndarray): Binary mask of shape (H, W) where 1 indicates ink and 0 indicates background.

    Returns:
        str: Space-delimited string of start positions and run lengths (e.g., '1 3 10 5').
    """
    # Flatten the image in row-major order (C-style)
    pixels = img.flatten()

    # Pad with zeros at the beginning and end to efficiently detect runs at the boundaries
    # We prepend and append 0 to handle cases where the run starts at index 0 or ends at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0 to 1 or 1 to 0)
    # np.where returns a tuple, we take the first element (array of indices)
    # We add 1 because the change happens *after* the index reported by where.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains [start1, end1, start2, end2, ...]
    # We want [start1, length1, start2, length2, ...]
    # Length = end - start
    # We update the odd indices (ends) to be lengths
    runs[1::2] -= runs[::2]

    # Convert to space-separated string
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary segmentation.
    The F0.5 score weights precision higher than recall.

    Args:
        preds (torch.Tensor): Predicted probabilities (0-1).
        targets (torch.Tensor): Ground truth binary masks (0 or 1).
        beta (float): Weight of precision in harmonic mean. Default 0.5.
        threshold (float): Threshold to binarize predictions. Default 0.5.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    # Formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)
    return score.item()


def dice_coef(preds, targets, threshold=0.5, epsilon=1e-7):
    """
    Calculates the Dice coefficient (F1 score).
    Dice = 2*TP / (2*TP + FP + FN)

    Args:
        preds (torch.Tensor): Predicted probabilities.
        targets (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to binarize predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tp = (preds_bin * targets_bin).sum()

    numerator = 2 * tp
    denominator = preds_bin.sum() + targets_bin.sum()

    score = numerator / (denominator + epsilon)
    return score.item()

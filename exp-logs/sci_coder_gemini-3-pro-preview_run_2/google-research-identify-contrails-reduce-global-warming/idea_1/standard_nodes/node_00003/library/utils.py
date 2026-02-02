import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Calculate the Dice coefficient.

    The formula is: 2 * |X intersect Y| / (|X| + |Y|)
    This function flattens the inputs to compute the global Dice score
    for the provided batch or array.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted binary mask (0 or 1).
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask (0 or 1).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The computed Dice coefficient.
    """
    if torch.is_tensor(y_pred):
        y_pred = y_pred.view(-1).float()
        y_true = y_true.view(-1).float()
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection) / (y_pred.sum() + y_true.sum() + smooth)
        return dice.item()
    else:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        intersection = np.sum(y_pred * y_true)
        dice = (2.0 * intersection) / (np.sum(y_pred) + np.sum(y_true) + smooth)
        return dice


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.

    The competition specifies: "The pixels are numbered from top to bottom,
    then left to right". This corresponds to flattening the array in
    column-major order (Fortran-style).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Run-length encoded string consisting of start pixel and length pairs,
             or '-' if the mask is empty.
    """
    # Flatten in column-major order (Fortran-style) to match 'top-bottom, left-right' indexing
    pixels = mask.flatten(order="F")

    # If mask is empty, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains start indices of segments.
    # Even indices (0, 2, ...) are starts of 1s, odd indices are starts of 0s.
    # We want lengths for the segments of 1s.
    # Replace end index with length: length = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

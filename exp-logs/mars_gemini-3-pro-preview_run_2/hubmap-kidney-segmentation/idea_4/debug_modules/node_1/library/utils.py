import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Run-length encoding for a binary mask.

    Args:
        img (np.ndarray): Binary mask where 1 indicates the object and 0 background.

    Returns:
        str: Run-length encoded string formatted as 'start length start length ...'.
    """
    # Flatten column-wise (Fortran-style) as required by the competition
    pixels = img.flatten(order="F")
    # Pad with 0s to detect starts and ends of runs at edges
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice Coefficient for PyTorch tensors.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The calculated Dice coefficient.
    """
    # Ensure inputs are float
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Flatten predictions and targets to compute global Dice
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = torch.sum(y_pred * y_true)
    dice = (2.0 * intersection + smooth) / (
        torch.sum(y_pred) + torch.sum(y_true) + smooth
    )

    return dice


def get_score(y_pred, y_true, threshold=0.5):
    """
    Calculate the Dice score for validation after thresholding.

    Args:
        y_pred (torch.Tensor): Predicted logits or probabilities.
        y_true (torch.Tensor): Ground truth mask.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        float: The Dice score.
    """
    # Apply sigmoid if predictions appear to be logits (outside [0, 1])
    if y_pred.min() < 0 or y_pred.max() > 1:
        y_pred = torch.sigmoid(y_pred)

    # Binarize predictions
    y_pred_bin = (y_pred > threshold).float()

    # Calculate Dice coefficient
    score = dice_coef(y_pred_bin, y_true)

    return score.item()

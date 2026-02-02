import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) as per competition format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates mask and 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if mask is empty.
             Pixels are numbered from top to bottom, then left to right.
    """
    # Flatten column-wise (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to efficiently detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # If no runs found (empty mask), return '-'
    if len(runs) == 0:
        return "-"

    # Calculate lengths: runs[1::2] are end indices, runs[::2] are start indices
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef_metric(y_pred, y_true, threshold=Config.THRESHOLD, smooth=1e-6):
    """
    Calculates the Dice Coefficient between predicted probabilities and ground truth masks.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (0-1).
        y_true (torch.Tensor or np.ndarray): Ground truth binary masks (0 or 1).
        threshold (float): Threshold to binarize predictions. Defaults to Config.THRESHOLD.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    # Flatten arrays to compute the metric over the entire provided volume (Batch-level or Global)
    y_pred_flat = y_pred_bin.flatten()
    y_true_flat = y_true_bin.flatten()

    # Calculate Intersection and Union
    intersection = np.sum(y_pred_flat * y_true_flat)
    union = np.sum(y_pred_flat) + np.sum(y_true_flat)

    # Compute Dice Coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice

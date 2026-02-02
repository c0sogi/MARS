import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def dice_coefficient(y_pred, y_true, threshold=0.5, smooth=1e-6):
    """
    Computes the Dice Coefficient for binary segmentation tasks.

    This function binarizes the predicted probabilities using the provided threshold
    and calculates the global Dice score for the input batch.

    Args:
        y_pred (torch.Tensor): Predicted probabilities with values in [0, 1].
        y_true (torch.Tensor): Ground truth binary masks (0 or 1).
        threshold (float): Threshold to convert probabilities to binary mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The computed Dice coefficient.
    """
    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # Binarize predictions
    pred_mask = (y_pred > threshold).float()

    # Flatten inputs to 1D to compute global dice over the batch
    # This handles various input shapes (e.g., [B, 1, H, W] vs [B, H, W]) robustly
    pred_flat = pred_mask.contiguous().view(-1)
    true_flat = y_true.contiguous().view(-1).float()

    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: A space-delimited string of 'start length' pairs, or '-' if the mask is empty.
    """
    # Flatten in column-major order (Fortran-style) to match the "top-to-bottom, left-to-right" requirement
    pixels = mask.flatten(order="F")

    # Check for empty predictions
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to correctly detect runs at the boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' currently holds the start indices of value changes.
    # Even indices (0, 2, ...) are starts of 1s (since we padded with 0).
    # Odd indices (1, 3, ...) are ends of 1s (starts of 0s).

    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

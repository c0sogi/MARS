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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The mask is flattened in column-major order (Fortran-style), which corresponds
    to numbering pixels from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) or (H, W, 1).
                           Values should be 0 (background) or 1 (foreground).

    Returns:
        str: Space-delimited string of RLE pairs (start length), or '-' if empty.
    """
    # Ensure mask is 2D or flattenable
    pixels = mask.flatten(order="F")

    # Check for empty mask
    if np.sum(pixels) == 0:
        return "-"

    # Add zero padding at start and end to detect transitions correctly
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef_metric(y_pred, y_true, threshold=Config.THRESHOLD, smooth=1e-6):
    """
    Calculates the Global Dice Coefficient for a batch of predictions.

    The metric treats the entire batch as a single volume (Global Dice),
    consistent with the competition metric.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
        y_true (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The calculated Dice coefficient.
    """
    # Ensure inputs are on the same device and float
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Binarize predictions
    y_pred = (y_pred > threshold).float()

    # Flatten the tensors to treat the batch as a global set
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Calculate intersection and union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    dice = (2.0 * intersection) / (union + smooth)

    return dice.item()

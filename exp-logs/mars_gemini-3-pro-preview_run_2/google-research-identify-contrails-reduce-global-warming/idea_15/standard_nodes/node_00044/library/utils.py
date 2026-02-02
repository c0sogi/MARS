import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The mask is flattened in column-major order (top to bottom, then left to right).
    Empty predictions are marked with '-'.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).
                           Values should be 0 (background) or 1 (foreground).

    Returns:
        str: Space-delimited list of pairs (start, length) or '-'.
    """
    # Flatten in column-major order as per competition requirement
    pixels = mask.flatten(order="F")

    # If no pixels are masked, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Prepend and append 0 to detect transitions
    # pixels is expected to be binary (0 or 1)
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef_batch(y_pred, y_true, threshold=0.5):
    """
    Calculates the Dice coefficient across the entire flattened batch.

    This metric treats the batch as a single volume, which aligns with the
    Global Dice metric used in the competition.

    Args:
        y_pred (torch.Tensor): Predicted probabilities.
        y_true (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.

    Returns:
        float: The global Dice coefficient for the batch.
    """
    # Flatten predictions and targets
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Binarize predictions
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()

    # Compute intersection and union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    # Handle edge case where both sets are empty (perfect prediction of background)
    if union == 0:
        return 1.0

    dice = (2.0 * intersection) / union

    return dice.item()

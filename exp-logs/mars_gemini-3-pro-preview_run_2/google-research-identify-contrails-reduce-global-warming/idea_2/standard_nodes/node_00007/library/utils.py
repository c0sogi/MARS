import os
import random
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The pixels are numbered from top to bottom, then left to right:
    1 is pixel (1,1), 2 is pixel (2,1), etc. (Column-major order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           0 indicates background, 1 indicates object.

    Returns:
        str: Space-delimited list of pairs (start_pixel, length).
             Returns '-' if the mask is empty.
    """
    # Flatten in column-major order (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # If mask is empty or contains no positive pixels
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to detect transitions correctly
    # Concatenate expects a sequence of arrays
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    # np.where returns a tuple, we take the first element
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs array structure: [start1, end1, start2, end2, ...]
    # The 'end' indices are exclusive in the run, but here they mark the first 0 after a 1.
    # Since we want length, length = end - start.
    # We modify the array in place to store lengths at the odd indices.
    runs[1::2] -= runs[0::2]

    # Convert to string space-delimited
    return " ".join(str(x) for x in runs)


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Scalar Dice coefficient.
    """
    # Flatten tensors to 1D
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    # Compute intersection and union
    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

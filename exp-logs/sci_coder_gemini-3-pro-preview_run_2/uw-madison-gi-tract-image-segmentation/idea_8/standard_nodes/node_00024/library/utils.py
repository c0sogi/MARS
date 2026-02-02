import os
import random
import numpy as np
import torch
import pandas as pd
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-wise (Fortran-style) as per competition spec
    pixels = img.flatten(order="F")

    # Pad with 0s at start and end to detect transitions efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains start indices (1-based) and end indices.
    # We calculate lengths by subtracting start indices from end indices.
    # runs[0] is start, runs[1] is end (exclusive in 0-based, or next start)
    # Since we want length, we do runs[1] - runs[0].
    # We modify the array in-place to store lengths in the odd positions.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Extract starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise to match encoding order
    return img.reshape(shape, order="F")


def compute_dice_score(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient between a predicted mask and a ground truth mask.

    Args:
        y_pred (np.ndarray): Predicted binary mask.
        y_true (np.ndarray): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()

    intersection = np.sum(y_pred_f * y_true_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_pred_f) + np.sum(y_true_f) + smooth
    )

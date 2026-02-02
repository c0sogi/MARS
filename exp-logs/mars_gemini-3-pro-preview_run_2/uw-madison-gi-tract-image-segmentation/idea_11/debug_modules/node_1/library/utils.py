import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right (Fortran/column-major).

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or mask_rle == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient between two binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def hausdorff_3d(y_true, y_pred):
    """
    Calculates the 3D Hausdorff distance between two 3D binary volumes.
    Coordinates are normalized by image size (H, W) to create a bounded score for spatial dimensions.
    Slice depth (Z) is treated with a step size of 1.

    Args:
        y_true (np.ndarray): Ground truth 3D mask of shape (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D mask of shape (Depth, Height, Width).

    Returns:
        float: The directed Hausdorff distance (max of both directions).
    """
    # Check for empty masks
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)

    if true_sum == 0 and pred_sum == 0:
        return 0.0
    if true_sum == 0 or pred_sum == 0:
        return np.inf

    # Get coordinates of non-zero pixels
    # shape is (Depth, Height, Width)
    d, h, w = y_true.shape

    # Get indices (z, y, x)
    z_true, y_true_idx, x_true_idx = np.where(y_true > 0)
    z_pred, y_pred_idx, x_pred_idx = np.where(y_pred > 0)

    # Normalize coordinates: y/H, x/W. Z is kept as slice index (depth 1 per slice).
    coords_true = np.stack([z_true, y_true_idx / h, x_true_idx / w], axis=1)
    coords_pred = np.stack([z_pred, y_pred_idx / h, x_pred_idx / w], axis=1)

    d1 = directed_hausdorff(coords_true, coords_pred)[0]
    d2 = directed_hausdorff(coords_pred, coords_true)[0]

    return max(d1, d2)

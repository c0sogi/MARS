import numpy as np
import torch
import os
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from library.config import Config


def set_seed(seed=42):
    """
    Sets seeds for reproducibility across random, numpy, and torch.

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

    The mask is flattened in column-major order (Fortran-style).
    The output is a space-delimited string of 'start length' pairs.
    Indices are 1-based.

    Args:
        img (np.ndarray): Binary mask of shape (Height, Width).

    Returns:
        str: RLE string.
    """
    # Ensure image is binary and flatten column-major
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask of shape (Height, Width).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def compute_dice_score(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient between predicted and ground truth masks.

    Args:
        y_pred (np.ndarray): Predicted binary mask.
        y_true (np.ndarray): Ground truth binary mask.
        smooth (float): Smoothing factor.

    Returns:
        float: Dice coefficient.
    """
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_pred_f * y_true_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_pred_f) + np.sum(y_true_f) + smooth
    )


def compute_hausdorff_distance(y_pred, y_true, shape):
    """
    Computes the 3D Hausdorff distance between two segmentation volumes.

    Pixel locations are normalized by image size (Height, Width) for the spatial
    dimensions. The Z dimension (slice depth) is treated as 1 unit per slice.

    Args:
        y_pred (np.ndarray): Predicted 3D binary volume (Depth, Height, Width).
        y_true (np.ndarray): Ground truth 3D binary volume (Depth, Height, Width).
        shape (tuple): The (Height, Width) of the 2D slices for normalization.

    Returns:
        float: The Hausdorff distance.
    """
    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)

    # Get coordinates of active pixels (z, y, x)
    pred_coords = np.argwhere(y_pred)
    true_coords = np.argwhere(y_true)

    # Handle empty masks
    if len(pred_coords) == 0 and len(true_coords) == 0:
        return 0.0
    if len(pred_coords) == 0 or len(true_coords) == 0:
        # Return a penalty if one is empty.
        # Since the metric is bounded 0-1 (via normalization), 1.0 is a reasonable max penalty.
        return 1.0

    # Convert to float for normalization
    p_coords = pred_coords.astype(np.float32)
    t_coords = true_coords.astype(np.float32)

    # Normalize Y and X coordinates (indices 1 and 2)
    H, W = shape
    p_coords[:, 1] /= H
    p_coords[:, 2] /= W
    t_coords[:, 1] /= H
    t_coords[:, 2] /= W

    # Z coordinate (index 0) remains as slice index (depth=1)

    # Compute directed Hausdorff distances using NearestNeighbors
    # d(A, B) = max(min_dist(a, B) for a in A)

    # 1. Fit on True, Query Pred -> d(Pred, True)
    nbrs_true = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(
        t_coords
    )
    dists_p_to_t, _ = nbrs_true.kneighbors(p_coords)
    d_pred_true = np.max(dists_p_to_t)

    # 2. Fit on Pred, Query True -> d(True, Pred)
    nbrs_pred = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(
        p_coords
    )
    dists_t_to_p, _ = nbrs_pred.kneighbors(t_coords)
    d_true_pred = np.max(dists_t_to_p)

    return max(d_pred_true, d_true_pred)

import numpy as np
import torch
import os
import random
import cv2
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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
    Args:
        img (np.ndarray): Binary mask, 2D or 3D.
    Returns:
        str: RLE encoded string.
    """
    # Flatten column-wise (Fortran style) as per competition spec
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Args:
        mask_rle (str): RLE encoded string.
        shape (tuple): (height, width) of the mask.
    Returns:
        np.ndarray: Binary mask.
    """
    if str(mask_rle) == "nan" or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calculate_dice(pred_mask, gt_mask):
    """
    Calculates Dice coefficient.
    Dice = 2 * |X n Y| / (|X| + |Y|)
    Returns 0 if both are empty.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())


def calculate_hausdorff_3d(pred_mask, gt_mask):
    """
    Calculates 3D Hausdorff distance with normalized coordinates.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # Check for empty masks
    pred_empty = pred_mask.sum() == 0
    gt_empty = gt_mask.sum() == 0

    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return 1.0

    # Get dimensions
    d, h, w = pred_mask.shape

    # Get coordinates of active pixels
    pred_points = np.argwhere(pred_mask)
    gt_points = np.argwhere(gt_mask)

    # Normalize coordinates to unit cube [0, 1]
    # This addresses "normalized by image size" and "bounded 0-1 score"
    pred_points = pred_points.astype(float)
    gt_points = gt_points.astype(float)

    pred_points[:, 0] /= d
    pred_points[:, 1] /= h
    pred_points[:, 2] /= w

    gt_points[:, 0] /= d
    gt_points[:, 1] /= h
    gt_points[:, 2] /= w

    # Compute directed Hausdorff distances
    # directed_hausdorff returns (distance, index_1, index_2)
    d_pg = directed_hausdorff(pred_points, gt_points)[0]
    d_gp = directed_hausdorff(gt_points, pred_points)[0]

    return max(d_pg, d_gp)


def compute_metrics(pred_vol, gt_vol):
    """
    Computes both Dice and Hausdorff for a 3D volume.
    Args:
        pred_vol: 3D array (D, H, W)
        gt_vol: 3D array (D, H, W)
    Returns:
        dict: {'dice': float, 'hausdorff': float}
    """
    dice = calculate_dice(pred_vol, gt_vol)
    hd = calculate_hausdorff_3d(pred_vol, gt_vol)
    return {"dice": dice, "hausdorff": hd}

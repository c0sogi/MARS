import os
import random
import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: RLE string 'start length start length ...'
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if str(mask_rle) == "nan" or mask_rle == "" or mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coef(y_true, y_pred):
    """
    Calculates the Dice coefficient.
    Formula: 2 * |X n Y| / (|X| + |Y|)
    Defined to be 0 when both X and Y are empty.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.

    Returns:
        float: Dice coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    sum_pixels = np.sum(y_true_f) + np.sum(y_pred_f)

    if sum_pixels == 0:
        return 0.0

    return (2.0 * intersection) / sum_pixels


def hausdorff_distance_3d(y_true, y_pred):
    """
    Calculates the 3D Hausdorff distance between two binary volumes.
    Pixel locations are normalized by image size to create a bounded 0-1 score.

    Args:
        y_true (np.ndarray): Ground truth 3D binary volume (D, H, W).
        y_pred (np.ndarray): Predicted 3D binary volume (D, H, W).

    Returns:
        float: Normalized Hausdorff distance.
    """
    # Get coordinates of non-zero pixels
    true_points = np.argwhere(y_true > 0)
    pred_points = np.argwhere(y_pred > 0)

    # Handle empty cases
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        return 1.0

    # Normalize coordinates by volume dimensions (D, H, W)
    shape = np.array(y_true.shape, dtype=float)
    true_points_norm = true_points / shape
    pred_points_norm = pred_points / shape

    # Calculate directed Hausdorff distances
    d_ab = directed_hausdorff(true_points_norm, pred_points_norm)[0]
    d_ba = directed_hausdorff(pred_points_norm, true_points_norm)[0]

    return max(d_ab, d_ba)


def keep_largest_component_3d(mask_3d):
    """
    Post-processing to keep only the largest connected component in a 3D volume.

    Args:
        mask_3d (np.ndarray): 3D binary mask (D, H, W).

    Returns:
        np.ndarray: Processed 3D binary mask.
    """
    mask_3d = mask_3d.astype(bool)
    labeled_mask, num_features = label(mask_3d)

    if num_features == 0:
        return mask_3d.astype(np.uint8)

    # Count pixels in each component
    # bincount is efficient; index 0 is background
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0  # Ignore background

    if counts.max() == 0:
        return mask_3d.astype(np.uint8)

    # Find label of largest component
    max_label = counts.argmax()

    # Create mask for largest component
    result = (labeled_mask == max_label).astype(np.uint8)
    return result

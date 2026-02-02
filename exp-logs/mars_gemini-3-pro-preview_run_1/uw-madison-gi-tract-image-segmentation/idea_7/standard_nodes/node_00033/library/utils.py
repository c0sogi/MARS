import os
import numpy as np
import cv2
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, label
from library.config import CLASSES


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    Pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img (np.array): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.array: Binary mask of shape (height, width).
    """
    if not isinstance(mask_rle, str) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def percentile_normalize(img, lower=1.0, upper=99.0):
    """
    Applies robust min-max normalization based on percentiles.
    Clips values to [p_lower, p_upper] and scales to [0, 1].

    Args:
        img (np.array): Input image.
        lower (float): Lower percentile threshold.
        upper (float): Upper percentile threshold.

    Returns:
        np.array: Normalized image (float32) in range [0, 1].
    """
    img = img.astype(np.float32)
    p_lower = np.percentile(img, lower)
    p_upper = np.percentile(img, upper)

    img = np.clip(img, p_lower, p_upper)

    if p_upper > p_lower:
        img = (img - p_lower) / (p_upper - p_lower)
    else:
        img = np.zeros_like(img)

    return img


def keep_largest_component(mask_3d):
    """
    Post-processing step to retain only the largest connected component in a 3D volume.
    Used to remove small false-positive noise.

    Args:
        mask_3d (np.array): 3D binary mask of shape (D, H, W).

    Returns:
        np.array: Processed 3D binary mask.
    """
    mask_3d = mask_3d.astype(np.uint8)
    labeled_mask, num_features = label(mask_3d)

    if num_features == 0:
        return mask_3d

    # Identify the largest region by volume
    # bincount returns counts for labels 0..N. Index 0 is background.
    component_sizes = np.bincount(labeled_mask.ravel())

    if len(component_sizes) < 2:
        return mask_3d

    # Find label with largest size (ignoring background at index 0)
    largest_label = component_sizes[1:].argmax() + 1

    # Create mask containing only the largest region
    new_mask = (labeled_mask == largest_label).astype(np.uint8)
    return new_mask


def compute_dice(y_pred, y_true):
    """
    Computes the Dice coefficient between predicted and ground truth masks.

    Args:
        y_pred (np.array): Predicted binary mask.
        y_true (np.array): Ground truth binary mask.

    Returns:
        float: Dice coefficient. Returns 0.0 if both masks are empty.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    pred_sum = y_pred.sum()
    true_sum = y_true.sum()

    # Specific requirement: Dice is 0 if both are empty
    if pred_sum == 0 and true_sum == 0:
        return 0.0

    intersection = (y_pred * y_true).sum()
    union = pred_sum + true_sum

    if union == 0:
        return 0.0

    return (2.0 * intersection) / union


def get_surface_points(mask):
    """
    Extracts the surface points of a 3D binary mask using morphological erosion.
    Used to speed up Hausdorff distance calculation.
    """
    # 3D 6-connectivity structure
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = True
    struct[0, 1, 1] = True
    struct[2, 1, 1] = True
    struct[1, 0, 1] = True
    struct[1, 2, 1] = True
    struct[1, 1, 0] = True
    struct[1, 1, 2] = True

    eroded = binary_erosion(mask, structure=struct)
    border = mask ^ eroded
    return np.argwhere(border).astype(np.float32)


def compute_hausdorff_3d(pred_vol, true_vol, spacing=None):
    """
    Computes the 3D Hausdorff distance.
    Coordinates are normalized by image dimensions (x/W, y/H) while Z is kept as slice index.

    Args:
        pred_vol (np.array): Predicted 3D mask (D, H, W).
        true_vol (np.array): Ground truth 3D mask (D, H, W).
        spacing (tuple): Placeholder for compatibility, not used for this specific metric definition.

    Returns:
        float: Hausdorff distance.
    """
    # Extract surface points to optimize calculation
    pred_points = get_surface_points(pred_vol > 0)
    true_points = get_surface_points(true_vol > 0)

    # Handle empty cases
    if len(pred_points) == 0 and len(true_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(true_points) == 0:
        return 1.0  # Return 1.0 (bounded max) if one is empty

    # Normalize coordinates
    # Shape is (D, H, W) -> Points are (z, y, x)
    d, h, w = pred_vol.shape

    # Normalize z, y, and x by image dimensions
    pred_points[:, 0] /= d
    pred_points[:, 1] /= h
    pred_points[:, 2] /= w

    true_points[:, 0] /= d
    true_points[:, 1] /= h
    true_points[:, 2] /= w

    # Calculate directed Hausdorff distances
    d_ab = directed_hausdorff(pred_points, true_points)[0]
    d_ba = directed_hausdorff(true_points, pred_points)[0]

    return max(d_ab, d_ba)

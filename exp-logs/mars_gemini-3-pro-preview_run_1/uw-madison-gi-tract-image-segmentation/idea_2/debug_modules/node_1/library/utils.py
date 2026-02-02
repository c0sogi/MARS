import os
import random
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label, sum as ndi_sum


def seed_everything(seed=42):
    """
    Sets the seed for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    Pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img: numpy array, 1 - mask, 0 - background
    Returns:
        String containing pairs of values (start length)
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
        mask_rle: run-length string (start length ...)
        shape: (height, width) of array to return
    Returns:
        Mask array, 1 - mask, 0 - background
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def compute_dice_score(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice Coefficient.

    Args:
        y_pred: Predicted binary mask (numpy array).
        y_true: Ground truth binary mask (numpy array).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice score (float).
    """
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_pred_f * y_true_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_pred_f) + np.sum(y_true_f) + smooth
    )


def compute_hausdorff_3d(y_pred, y_true):
    """
    Computes the 3D Hausdorff Distance.
    Coordinates are normalized by image size (H, W) for x and y.
    Z is treated as slice index.

    Args:
        y_pred: Predicted 3D mask (D, H, W).
        y_true: Ground truth 3D mask (D, H, W).
    Returns:
        Hausdorff distance (float).
    """
    # Find indices of non-zero pixels
    # np.argwhere returns (z, y, x) for (D, H, W) input
    pred_points = np.argwhere(y_pred > 0)
    true_points = np.argwhere(y_true > 0)

    # Handle empty masks
    if len(pred_points) == 0 and len(true_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(true_points) == 0:
        return 1.0  # Maximum penalty

    d, h, w = y_pred.shape

    # Convert to float for normalization
    pred_points = pred_points.astype(float)
    true_points = true_points.astype(float)

    # Normalize Y (index 1) and X (index 2) by image dimensions
    # Z (index 0) remains as slice index
    pred_points[:, 1] /= h
    pred_points[:, 2] /= w
    true_points[:, 1] /= h
    true_points[:, 2] /= w

    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(pred_points, true_points)[0]
    d2 = directed_hausdorff(true_points, pred_points)[0]

    return max(d1, d2)


def keep_largest_component_3d(mask):
    """
    Keeps only the largest connected component in a 3D binary mask.

    Args:
        mask: 3D numpy array (binary).
    Returns:
        Processed mask with only the largest component.
    """
    mask = mask.astype(np.uint8)
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return mask

    # Calculate size of each component
    # label 0 is background, so we iterate from 1 to num_features
    component_sizes = ndi_sum(mask, labeled_mask, range(1, num_features + 1))

    if np.isscalar(component_sizes):
        component_sizes = [component_sizes]

    if len(component_sizes) == 0:
        return mask

    # Identify label of largest component
    max_label = np.argmax(component_sizes) + 1

    # Create new mask retaining only the largest component
    new_mask = (labeled_mask == max_label).astype(np.uint8)
    return new_mask


def compute_metrics(y_pred, y_true):
    """
    Computes the combined competition metric.
    Metric = 0.4 * Dice + 0.6 * (1 - Hausdorff)

    Args:
        y_pred: Predicted 3D mask.
        y_true: Ground truth 3D mask.
    Returns:
        Dictionary containing 'dice', 'hausdorff', and 'score'.
    """
    dice = compute_dice_score(y_pred, y_true)
    hd = compute_hausdorff_3d(y_pred, y_true)

    # Score calculation: 1 - HD converts distance to a similarity score.
    # We clip the term (1 - HD) to be non-negative.
    score = 0.4 * dice + 0.6 * max(0.0, 1.0 - hd)

    return {"dice": dice, "hausdorff": hd, "score": score}

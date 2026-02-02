import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W).
                          Pixels are numbered from top to bottom, then left to right.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style)
    pixels = img.flatten(order="F")

    # Pad with zeros to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def calculate_dice(y_true, y_pred):
    """
    Calculates the Dice coefficient.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        float: Dice coefficient.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    sum_true = y_true.sum()
    sum_pred = y_pred.sum()

    # As per task description: Dice is 0 when both X and Y are empty
    if sum_true == 0 and sum_pred == 0:
        return 0.0

    return (2.0 * intersection) / (sum_true + sum_pred)


def calculate_hausdorff(y_true_3d, y_pred_3d):
    """
    Calculates the 3D Hausdorff distance.
    Coordinates are normalized by image dimensions (H, W). Z is slice index (depth=1).

    Args:
        y_true_3d (np.ndarray): Ground truth 3D mask (D, H, W).
        y_pred_3d (np.ndarray): Predicted 3D mask (D, H, W).

    Returns:
        float: Hausdorff distance.
    """
    true_points = np.argwhere(y_true_3d > 0)
    pred_points = np.argwhere(y_pred_3d > 0)

    # Handle empty cases
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        # Return a large penalty value if one is empty
        return 100.0

    D, H, W = y_true_3d.shape

    true_points = true_points.astype(float)
    pred_points = pred_points.astype(float)

    # Normalize Y (index 1) and X (index 2)
    # Z (index 0) remains as slice index
    true_points[:, 1] /= H
    true_points[:, 2] /= W

    pred_points[:, 1] /= H
    pred_points[:, 2] /= W

    # Calculate directed Hausdorff distances
    d_ab = directed_hausdorff(true_points, pred_points)[0]
    d_ba = directed_hausdorff(pred_points, true_points)[0]

    return max(d_ab, d_ba)


def keep_largest_component(mask):
    """
    Keeps only the largest connected component in the mask.

    Args:
        mask (np.ndarray): Binary mask (2D or 3D).

    Returns:
        np.ndarray: Mask with only the largest component.
    """
    mask = mask.astype(bool)
    labeled_mask, num_labels = ndimage.label(mask)

    if num_labels == 0:
        return mask.astype(np.uint8)

    # Calculate size of each component (label 0 is background)
    sizes = ndimage.sum(mask, labeled_mask, range(num_labels + 1))

    # Find label with max size (ignoring background at index 0)
    largest_label = sizes[1:].argmax() + 1

    return (labeled_mask == largest_label).astype(np.uint8)


def get_gaussian_weight_map(shape, sigma_scale=0.25):
    """
    Generates a Gaussian weight map for sliding window blending.

    Args:
        shape (tuple): (H, W) dimensions.
        sigma_scale (float): Standard deviation relative to size.

    Returns:
        np.ndarray: Weight map in [0, 1].
    """
    H, W = shape
    center_y, center_x = H // 2, W // 2

    y, x = np.ogrid[:H, :W]

    sigma_y = H * sigma_scale
    sigma_x = W * sigma_scale

    gauss = np.exp(
        -(
            (x - center_x) ** 2 / (2 * sigma_x**2)
            + (y - center_y) ** 2 / (2 * sigma_y**2)
        )
    )

    if gauss.max() > 0:
        gauss /= gauss.max()

    return gauss.astype(np.float32)

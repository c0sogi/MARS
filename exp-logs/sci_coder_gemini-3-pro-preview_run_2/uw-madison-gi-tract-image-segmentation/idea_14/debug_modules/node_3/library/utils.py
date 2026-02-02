import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The output is a space-delimited string of pairs (start, length).
    Pixels are numbered from top to bottom, then left to right (column-major).
    Indices are 1-based.

    Args:
        img (np.ndarray): Binary mask of shape (height, width).
                          0 indicates background, 1 indicates object.

    Returns:
        str: RLE string.
    """
    # Flatten column-wise (Fortran-style)
    pixels = img.flatten(order="F")

    # Add zero at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array now contains alternating start and end indices
    # runs[0] is start of first run of 1s
    # runs[1] is start of first run of 0s (which marks end of 1s)
    # Length is runs[1] - runs[0]
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask (uint8) of shape `shape`.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape to 2D (Fortran-style to match encoding)
    return img.reshape(shape, order="F")


def load_image(path):
    """
    Loads a 16-bit PNG image and normalizes it to the [0, 1] range.

    Args:
        path (str): File path to the image.

    Returns:
        np.ndarray: Normalized image as float32.
    """
    # Load 16-bit image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image at {path}")

    # Convert to float32
    img = img.astype(np.float32)

    # Min-Max Normalization per image
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


def calculate_dice(y_true, y_pred):
    """
    Calculates the Dice coefficient between two binary masks.

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
    cardinality = np.sum(y_true_f) + np.sum(y_pred_f)

    if cardinality == 0:
        return 0.0

    return (2.0 * intersection) / cardinality


def calculate_hausdorff_3d(y_true, y_pred):
    """
    Calculates the 3D Hausdorff distance between two binary volumes.

    Coordinates (y, x) are normalized by the image height and width respectively.
    The Z-coordinate (slice depth) is treated as having unit distance (1.0).

    Args:
        y_true (np.ndarray): Ground truth 3D volume (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D volume (Depth, Height, Width).

    Returns:
        float: The directed Hausdorff distance (max of d(A,B) and d(B,A)).
               Returns 0.0 if both volumes are empty.
               Returns 100.0 if only one volume is empty (penalty).
    """
    # Extract coordinates of non-zero pixels: (z, y, x)
    true_points = np.argwhere(y_true > 0)
    pred_points = np.argwhere(y_pred > 0)

    len_true = len(true_points)
    len_pred = len(pred_points)

    # Handle empty cases
    if len_true == 0 and len_pred == 0:
        return 0.0
    if len_true == 0 or len_pred == 0:
        # Return a large penalty distance if one is empty
        return 100.0

    # Get dimensions for normalization
    depth, height, width = y_true.shape

    # Convert to float for calculation
    true_points = true_points.astype(np.float32)
    pred_points = pred_points.astype(np.float32)

    # Normalize Y (index 1) and X (index 2) coordinates
    # Z (index 0) remains unscaled (slice depth = 1)
    true_points[:, 1] /= height
    true_points[:, 2] /= width

    pred_points[:, 1] /= height
    pred_points[:, 2] /= width

    # Calculate directed Hausdorff distances
    # directed_hausdorff returns (distance, index_1, index_2)
    d_ab = directed_hausdorff(true_points, pred_points)[0]
    d_ba = directed_hausdorff(pred_points, true_points)[0]

    return max(d_ab, d_ba)

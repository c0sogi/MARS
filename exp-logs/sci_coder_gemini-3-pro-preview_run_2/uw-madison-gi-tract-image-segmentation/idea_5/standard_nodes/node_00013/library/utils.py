import numpy as np
import pandas as pd
from scipy.ndimage import label, distance_transform_edt
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask to RLE format (Run-Length Encoding).
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten in column-major order
    pixels = mask.flatten(order="F")
    # Pad to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Convert end indices to lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask.
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


def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary masks.
    Formula: 2*|X n Y| / (|X| + |Y|)
    Defined to be 0 when both X and Y are empty.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        float: Dice score.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    sum_true = np.sum(y_true_f)
    sum_pred = np.sum(y_pred_f)

    # Specific condition from prompt
    if sum_true == 0 and sum_pred == 0:
        return 0.0

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (sum_true + sum_pred + 1e-8)


def hausdorff_distance_3d(y_true, y_pred):
    """
    Computes the 3D Hausdorff distance.
    Pixel locations are normalized by image size (H, W) to [0, 1].
    Slice depth (Z) is treated with spacing 1.0.

    Args:
        y_true (np.ndarray): Ground truth 3D volume (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D volume (Depth, Height, Width).

    Returns:
        float: Directed Hausdorff distance (max of d(A,B) and d(B,A)).
    """
    # Handle empty cases
    true_empty = np.sum(y_true) == 0
    pred_empty = np.sum(y_pred) == 0

    if true_empty and pred_empty:
        return 0.0
    if true_empty or pred_empty:
        # If one is empty, return a penalty (1.0 is max normalized spatial distance)
        return 1.0

    depth, height, width = y_true.shape
    # Spacing: Z=1/D, Y=1/H, X=1/W
    # This normalizes spatial dimensions to 0-1 range
    spacing = (1.0 / depth, 1.0 / height, 1.0 / width)

    # Compute Distance Transform
    # input is 1-mask because edt calculates distance to nearest 0
    dt_pred = distance_transform_edt(1 - y_pred, sampling=spacing)
    dt_true = distance_transform_edt(1 - y_true, sampling=spacing)

    # Directed Hausdorff A -> B: max(dt_B[a]) for a in A
    d_true_pred = np.max(dt_pred[y_true > 0])

    # Directed Hausdorff B -> A: max(dt_A[b]) for b in B
    d_pred_true = np.max(dt_true[y_pred > 0])

    return max(d_true_pred, d_pred_true)


def keep_largest_connected_component_3d(mask):
    """
    Post-processing: Keeps only the largest connected component in a 3D volume.
    Removes smaller noise artifacts.

    Args:
        mask (np.ndarray): 3D binary mask (Depth, Height, Width).

    Returns:
        np.ndarray: Cleaned 3D binary mask.
    """
    mask = mask.astype(np.uint8)
    # Label connected components (default 3x3x3 connectivity for 3D)
    labeled_mask, num_features = label(mask)

    if num_features <= 1:
        return mask

    # Count pixels in each component
    # bincount returns count for label 0 (background), 1, 2...
    counts = np.bincount(labeled_mask.flatten())

    # Identify largest component (ignoring background at index 0)
    # counts[1:] corresponds to labels 1..N
    if len(counts) > 1:
        largest_label = np.argmax(counts[1:]) + 1
        return (labeled_mask == largest_label).astype(np.uint8)

    return mask

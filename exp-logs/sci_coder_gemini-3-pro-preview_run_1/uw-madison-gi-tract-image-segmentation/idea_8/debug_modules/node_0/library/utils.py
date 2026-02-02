import os
import random
import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt, label
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    Pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten column-wise (Fortran style)
    pixels = img.flatten(order="F")
    # Pad to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Convert end indices to lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if not mask_rle or mask_rle == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # 1-based to 0-based indexing
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise
    return img.reshape(shape, order="F")


def dice_coef(y_true, y_pred):
    """
    Calculates the Dice coefficient.
    Formula: 2 * |X n Y| / (|X| + |Y|)

    Note: As per requirements, returns 0 if both X and Y are empty.

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


def hausdorff_3d_distance(y_true, y_pred):
    """
    Calculates the 3D Hausdorff distance with normalized pixel locations.

    Normalization:
    - Z-axis spacing is set to 1.0.
    - Y-axis (height) is normalized by image height (spacing = 1/H).
    - X-axis (width) is normalized by image width (spacing = 1/W).

    Args:
        y_true (np.ndarray): Ground truth 3D volume (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D volume (Depth, Height, Width).

    Returns:
        float: The directed Hausdorff distance. Returns 0.0 if both empty, 1.0 if one empty.
    """
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return 1.0

    depth, height, width = y_true.shape

    # Sampling defines the physical distance between adjacent pixels along each axis
    # Z=1.0, Y=1.0/Height, X=1.0/Width
    spacing = np.array([1.0, 1.0 / height, 1.0 / width])

    # Compute Euclidean Distance Transform (EDT)
    # edt computes distance from background (0) to nearest foreground (1)
    # We invert masks because we want distance from foreground points
    dt_true = distance_transform_edt(1 - y_true, sampling=spacing)
    dt_pred = distance_transform_edt(1 - y_pred, sampling=spacing)

    # Directed Hausdorff A -> B is max(dt_B[a] for a in A)
    coords_pred = y_pred.astype(bool)
    d_pred_true = np.max(dt_true[coords_pred]) if np.any(coords_pred) else 0.0

    # Directed Hausdorff B -> A is max(dt_A[b] for b in B)
    coords_true = y_true.astype(bool)
    d_true_pred = np.max(dt_pred[coords_true]) if np.any(coords_true) else 0.0

    return max(d_pred_true, d_true_pred)


def keep_largest_component(mask_3d):
    """
    Post-processing utility to keep only the largest connected component in a 3D mask.
    Removes small noise/artifacts.

    Args:
        mask_3d (np.ndarray): Binary 3D mask (Depth, Height, Width).

    Returns:
        np.ndarray: Processed binary 3D mask.
    """
    labeled_array, num_features = label(mask_3d)
    if num_features == 0:
        return mask_3d

    # Count pixels in each component
    sizes = np.bincount(labeled_array.ravel())
    # Ignore background (label 0)
    sizes[0] = 0

    if sizes.max() == 0:
        return mask_3d

    max_label = sizes.argmax()

    return (labeled_array == max_label).astype(np.uint8)

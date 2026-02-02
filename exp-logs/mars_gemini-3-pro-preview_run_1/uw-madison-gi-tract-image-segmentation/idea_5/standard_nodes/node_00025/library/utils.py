import os
import random
import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_decode(mask_rle, shape):
    """
    Decodes a run-length encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if not mask_rle or str(mask_rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-based indexing to 0-based
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran order (column-major) as per competition spec
    return img.reshape(shape, order="F")


def rle_encode(img):
    """
    Encodes a binary mask into a run-length encoded string.

    Args:
        img (np.ndarray): Binary mask of shape (height, width).

    Returns:
        str: RLE string.
    """
    # Flatten using Fortran order (column-major)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient between two binary masks.
    Defined as 0 if both sets are empty.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero (unused in empty case).

    Returns:
        float: Dice coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    cardinality = np.sum(y_true_f) + np.sum(y_pred_f)

    # Strict adherence to prompt: Dice is 0 when both X and Y are empty
    if cardinality == 0:
        return 0.0

    return (2.0 * intersection) / (cardinality + smooth)


def get_3d_hausdorff(y_true, y_pred):
    """
    Computes the 3D Hausdorff distance between two binary volumes.
    Coordinates are normalized by volume dimensions to create a bounded 0-1 score.

    Args:
        y_true (np.ndarray): Ground truth volume (Depth, Height, Width).
        y_pred (np.ndarray): Predicted volume (Depth, Height, Width).

    Returns:
        float: Normalized 3D Hausdorff distance.
    """
    true_points = np.argwhere(y_true > 0)
    pred_points = np.argwhere(y_pred > 0)

    # Handle edge cases for empty masks
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        return 1.0  # Maximum penalty in normalized space

    # Normalize coordinates by the shape of the volume
    # shape is (Depth, Height, Width)
    shape = np.array(y_true.shape, dtype=float)

    true_points_norm = true_points / shape
    pred_points_norm = pred_points / shape

    # Compute directed Hausdorff distances
    d_ab = directed_hausdorff(true_points_norm, pred_points_norm)[0]
    d_ba = directed_hausdorff(pred_points_norm, true_points_norm)[0]

    return max(d_ab, d_ba)


def keep_largest_connected_component_3d(volume, min_size=50):
    """
    Keeps only the largest connected component in a 3D binary volume.

    Args:
        volume (np.ndarray): Binary volume (Depth, Height, Width).
        min_size (int): Minimum size (in pixels) of the component to keep.

    Returns:
        np.ndarray: Processed binary volume containing only the largest component.
    """
    # Use default structure (connectivity=1, i.e., 6-neighbors in 3D)
    labeled_array, num_features = label(volume)

    if num_features == 0:
        return volume

    # Count pixels per label (index 0 is background)
    sizes = np.bincount(labeled_array.ravel())

    # If only background exists
    if len(sizes) <= 1:
        return np.zeros_like(volume)

    # Identify largest component (excluding background at index 0)
    largest_label = sizes[1:].argmax() + 1

    # Check if largest component meets minimum size requirement
    if sizes[largest_label] < min_size:
        return np.zeros_like(volume)

    # Create mask for the largest component
    output = (labeled_array == largest_label).astype(np.uint8)

    return output

import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label, generate_binary_structure, binary_erosion

from library.config import Config


def set_seed(seed=Config.SEED):
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


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-major
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def compute_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Coefficient between ground truth and prediction.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth mask.
        y_pred (np.ndarray or torch.Tensor): Predicted mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def get_surface_points(mask):
    """
    Extracts surface points from a 3D mask to optimize Hausdorff calculation.
    """
    # 3D connectivity structure
    kernel = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=kernel)
    boundary = mask ^ eroded
    return np.argwhere(boundary)


def compute_hausdorff_distance(y_true, y_pred):
    """
    Computes the normalized 3D Hausdorff Distance.
    Coordinates are normalized by the image dimensions (Depth, Height, Width)
    to create a bounded score.

    Args:
        y_true (np.ndarray): 3D Ground truth mask (Depth, Height, Width).
        y_pred (np.ndarray): 3D Predicted mask (Depth, Height, Width).

    Returns:
        float: Normalized Hausdorff distance.
    """
    # Handle empty cases
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)

    if true_sum == 0 and pred_sum == 0:
        return 0.0
    if true_sum == 0 or pred_sum == 0:
        return 1.0  # Maximum penalty for empty vs non-empty mismatch

    # Optimization: Use only surface points
    true_points = get_surface_points(y_true).astype(float)
    pred_points = get_surface_points(y_pred).astype(float)

    if len(true_points) == 0 or len(pred_points) == 0:
        return 1.0

    # Normalize coordinates by dimensions (D, H, W)
    shape = y_true.shape
    depth, height, width = shape

    true_points[:, 0] /= depth
    true_points[:, 1] /= height
    true_points[:, 2] /= width

    pred_points[:, 0] /= depth
    pred_points[:, 1] /= height
    pred_points[:, 2] /= width

    # Compute symmetric Hausdorff distance
    d1 = directed_hausdorff(true_points, pred_points)[0]
    d2 = directed_hausdorff(pred_points, true_points)[0]

    return max(d1, d2)


def keep_largest_component(mask):
    """
    Retains only the largest connected component in the mask to remove noise.
    Supports both 2D (slice) and 3D (volume) masks.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    mask = mask.astype(np.uint8)

    # 2D Case
    if mask.ndim == 2:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels < 2:
            return mask

        # stats[:, 4] is area. Index 0 is background.
        # Find label with max area (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        return (labels == largest_label).astype(np.uint8)

    # 3D Case
    elif mask.ndim == 3:
        labeled_array, num_features = label(mask)
        if num_features == 0:
            return mask

        # Count pixels per label
        counts = np.bincount(labeled_array.ravel())
        counts[0] = 0  # Ignore background

        largest_label = counts.argmax()
        return (labeled_array == largest_label).astype(np.uint8)

    return mask


def load_or_create_data(filepath, create_func, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism. Loads data from filepath if it exists and
    load_cached_data is True. Otherwise, calls create_func(**kwargs) and saves the result.

    Args:
        filepath (str): Path to save/load the file.
        create_func (callable): Function to generate data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to create_func.

    Returns:
        Data loaded or created.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if load_cached_data and os.path.exists(filepath):
        print(f"Loading cached data from {filepath}")
        try:
            if filepath.endswith(".parquet"):
                return pd.read_parquet(filepath)
            elif filepath.endswith(".npy"):
                return np.load(filepath, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recreating...")

    print(f"Creating data -> {filepath}")
    data = create_func(**kwargs)

    if filepath.endswith(".parquet") and isinstance(data, pd.DataFrame):
        data.to_parquet(filepath)
    elif filepath.endswith(".npy"):
        np.save(filepath, data)

    return data

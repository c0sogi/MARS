import os
import random
import numpy as np
import torch
import cv2
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff


def remove_small_objects(mask, min_size=10):
    """
    Removes connected components smaller than min_size from a binary mask.
    Cite solution_lesson_node_00007: Instability of Hausdorff Distance without Morphological Post-Processing.
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    # stats[label, cv2.CC_STAT_AREA] is the area
    # Label 0 is background
    sizes = stats[1:, cv2.CC_STAT_AREA]
    keep_indices = np.where(sizes >= min_size)[0] + 1

    if len(keep_indices) == 0:
        return np.zeros_like(mask)

    return np.isin(labels, keep_indices).astype(np.uint8)


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
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img (numpy.ndarray): Binary mask (0 for background, 1 for object).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): Run-length encoded string.
        shape (tuple): Target shape (height, width) of the mask.

    Returns:
        numpy.ndarray: Binary mask with shape `shape`.
    """
    if mask_rle is None or not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient.
    CRITICAL: Returns 0.0 if both y_true and y_pred are empty, as per task rules.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    sum_true = np.sum(y_true_f)
    sum_pred = np.sum(y_pred_f)

    # Task specific rule: 0 when both are empty
    if sum_true == 0 and sum_pred == 0:
        return 0.0

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)


def hausdorff_distance_3d(y_true, y_pred, spacing=None):
    """
    Computes 3D Hausdorff distance between two binary volumes.
    Coordinates are normalized by volume dimensions (D, H, W) to [0, 1].

    Args:
        y_true: 3D numpy array (D, H, W)
        y_pred: 3D numpy array (D, H, W)
        spacing: tuple of (z_spacing, y_spacing, x_spacing). Ignored here as we normalize by shape.

    Returns:
        float: Normalized Hausdorff distance (0.0 to 1.0).
    """
    # Ensure binary
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5

    # Get coordinates of True pixels
    # argwhere returns (N, 3) array of [z, y, x] indices
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)

    # Handle empty cases
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        return 1.0  # Max distance if one is missing

    # Normalize coordinates to [0, 1]
    # Shape is (D, H, W)
    shape = np.array(y_true.shape, dtype=np.float32)

    # Normalize points: divide each column by corresponding dimension
    true_points_norm = true_points.astype(np.float32) / shape
    pred_points_norm = pred_points.astype(np.float32) / shape

    # Compute directed Hausdorff distances
    # directed_hausdorff returns (distance, index_1, index_2)
    d_ab = directed_hausdorff(true_points_norm, pred_points_norm)[0]
    d_ba = directed_hausdorff(pred_points_norm, true_points_norm)[0]

    return max(d_ab, d_ba)


def hausdorff_distance(y_true, y_pred):
    return hausdorff_distance_3d(y_true[np.newaxis, ...], y_pred[np.newaxis, ...])

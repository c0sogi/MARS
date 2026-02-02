import os
import random
import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        img (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran style)
    pixels = img.flatten(order="F")
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
        np.ndarray: Binary mask with the specified shape.
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


def compute_dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary masks.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        float: Dice coefficient. Returns 0.0 if both are empty (per task spec).
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    sum_true = np.sum(y_true_f)
    sum_pred = np.sum(y_pred_f)

    # Task Description: "The Dice coefficient is defined to be 0 when both X and Y are empty."
    if sum_true == 0 and sum_pred == 0:
        return 0.0

    return (2.0 * intersection) / (sum_true + sum_pred)


def compute_hausdorff_3d(y_true, y_pred, shape=None):
    """
    Computes the 3D Hausdorff distance between two binary volumes.
    Coordinates are normalized by the image size to create a bounded score.

    Args:
        y_true (np.ndarray): Ground truth 3D volume (D, H, W).
        y_pred (np.ndarray): Predicted 3D volume (D, H, W).
        shape (tuple, optional): Shape to use for normalization. Defaults to y_true.shape.

    Returns:
        float: The 3D Hausdorff distance on normalized coordinates.
    """
    if shape is None:
        shape = y_true.shape

    # Get coordinates of non-zero pixels (z, y, x)
    gt_coords = np.argwhere(y_true > 0)
    pred_coords = np.argwhere(y_pred > 0)

    # Handle empty cases
    len_gt = len(gt_coords)
    len_pred = len(pred_coords)

    if len_gt == 0 and len_pred == 0:
        return 0.0
    if len_gt == 0 or len_pred == 0:
        return 1.0  # Max penalty for bounded score context

    # Normalize coordinates by dimensions to map to [0, 1] range
    # shape is (D, H, W), coords are (z, y, x)
    gt_norm = gt_coords.astype(np.float32) / np.array(shape, dtype=np.float32)
    pred_norm = pred_coords.astype(np.float32) / np.array(shape, dtype=np.float32)

    # Compute directed Hausdorff distances
    # directed_hausdorff returns (max(min(d)), index_u, index_v)
    d_gt_pred = directed_hausdorff(gt_norm, pred_norm)[0]
    d_pred_gt = directed_hausdorff(pred_norm, gt_norm)[0]

    return max(d_gt_pred, d_pred_gt)


def apply_3d_cca(pred_volume):
    """
    Applies 3D Connected Component Analysis to retain only the largest connected component.

    Args:
        pred_volume (np.ndarray): Binary 3D volume.

    Returns:
        np.ndarray: Processed binary volume containing only the largest component.
    """
    # Use full connectivity (26-neighbors in 3D)
    structure = np.ones((3, 3, 3), dtype=int)
    labeled_array, num_features = label(pred_volume, structure=structure)

    if num_features == 0:
        return pred_volume

    # Calculate size of each component
    # bincount returns count of each label value. 0 is background.
    sizes = np.bincount(labeled_array.ravel())

    # If only background exists (should be caught by num_features check, but for safety)
    if len(sizes) <= 1:
        return pred_volume

    # Identify the label of the largest component (ignoring background at index 0)
    max_label = sizes[1:].argmax() + 1

    # Create mask for the largest component
    new_mask = (labeled_array == max_label).astype(np.uint8)

    return new_mask

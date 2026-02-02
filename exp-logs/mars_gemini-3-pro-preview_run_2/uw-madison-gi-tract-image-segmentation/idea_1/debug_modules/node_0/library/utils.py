import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, generate_binary_structure

from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The format is a space-delimited list of pairs (start_pixel, run_length).
    Pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        img (np.ndarray): Binary mask (0s and 1s) of shape (H, W).

    Returns:
        str: Space-delimited RLE string. Returns empty string if mask is empty.
    """
    # Flatten column-wise (Fortran-style) to match competition format
    pixels = img.flatten(order="F")

    # Pad with 0s to detect changes at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W) with dtype uint8.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to 2D using column-major order
    return img.reshape(shape, order="F")


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Coefficient between ground truth and prediction.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth mask.
        y_pred (np.ndarray or torch.Tensor): Predicted mask (binary or probability).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient score.
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
    Helper to extract surface points from a 3D volume to optimize Hausdorff calculation.
    Uses morphological operations to find the boundary of the object.
    """
    # Generate a 3D cross structure for connectivity
    struct = generate_binary_structure(3, 1)

    # Erode the mask to find the interior
    eroded = binary_erosion(mask, structure=struct)

    # XOR original and eroded to get the boundary (surface)
    surface = mask ^ eroded

    # Return coordinates of surface pixels
    return np.argwhere(surface > 0)


def hausdorff_3d(y_true, y_pred):
    """
    Computes the 3D Hausdorff Distance between two binary volumes.
    Normalizes X and Y coordinates by image dimensions. Z is treated with step 1.

    Args:
        y_true (np.ndarray): Ground truth 3D volume (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D volume (Depth, Height, Width).

    Returns:
        float: The directed Hausdorff distance.
               Returns 0.0 if both masks are empty.
               Returns 1.0 if only one mask is empty (max penalty).
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Check for empty masks
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)

    if true_sum == 0 and pred_sum == 0:
        return 0.0
    if true_sum == 0 or pred_sum == 0:
        return 1.0

    # Get surface points (z, y, x)
    true_points = get_surface_points(y_true).astype(float)
    pred_points = get_surface_points(y_pred).astype(float)

    # Normalize coordinates
    # Z (dim 0) is slice depth = 1 (not normalized per task description)
    # Y (dim 1) normalized by Height
    # X (dim 2) normalized by Width
    d, h, w = y_true.shape

    true_points[:, 1] /= h
    true_points[:, 2] /= w

    pred_points[:, 1] /= h
    pred_points[:, 2] /= w

    # Calculate directed Hausdorff distance
    # H(A, B) = max(h(A, B), h(B, A))
    d_ab = directed_hausdorff(true_points, pred_points)[0]
    d_ba = directed_hausdorff(pred_points, true_points)[0]

    return max(d_ab, d_ba)


def load_image(path):
    """
    Loads an image from the given path and normalizes it.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Normalized image array (float32) in range [0, 1].
    """
    # Handle relative paths if necessary
    if not os.path.exists(path):
        potential_path = os.path.join(Config.INPUT_DIR, path)
        if os.path.exists(potential_path):
            path = potential_path

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # Normalize based on bit depth
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0

    return img

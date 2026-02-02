import os
import cv2
import numpy as np
import torch
import random
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_image(path):
    """
    Loads an image from the given path, handling 16-bit depth and normalization.

    Args:
        path (str): Relative path to the image file from the input directory.

    Returns:
        np.ndarray: Normalized image of shape (H, W) with values in [0, 1].
    """
    full_path = os.path.join(Config.INPUT_DIR, path)

    # Load as unchanged to preserve 16-bit depth
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found at {full_path}")

    # Convert to float32 for processing
    img = img.astype(np.float32)

    # Min-Max Normalization per image to [0, 1]
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # Handle case where image is constant (e.g., all black)
        img = np.zeros_like(img)

    return img


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = img.flatten()
    # Add zeros at start and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
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
    # Handle NaN or empty values (common in pandas dataframes for empty masks)
    if mask_rle != mask_rle or mask_rle is None or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice Coefficient between prediction and ground truth.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Scalar Dice coefficient.
    """
    # Flatten tensors
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def hausdorff_distance_3d(y_pred, y_true):
    """
    Computes the 3D Hausdorff Distance between predicted and ground truth volumes.
    Coordinates are normalized by image dimensions (x/W, y/H) while Z is slice index.

    Args:
        y_pred (np.ndarray): Predicted binary volume (D, H, W).
        y_true (np.ndarray): Ground truth binary volume (D, H, W).

    Returns:
        float: The Hausdorff distance. Returns 0.0 if both are empty, 1.0 if one is empty.
    """
    # Ensure inputs are boolean/binary
    y_pred = y_pred > 0
    y_true = y_true > 0

    # Handle empty masks
    has_pred = np.any(y_pred)
    has_true = np.any(y_true)

    if not has_true and not has_pred:
        return 0.0
    if not has_true or not has_pred:
        return 1.0  # Max penalty for completely missing/hallucinated object

    # Get coordinates of non-zero pixels: (z, y, x)
    coords_pred = np.argwhere(y_pred).astype(np.float32)
    coords_true = np.argwhere(y_true).astype(np.float32)

    # Normalize coordinates
    # Shape is (Depth, Height, Width)
    depth, height, width = y_pred.shape

    # Normalize Y (index 1) and X (index 2) by image dimensions
    # Z (index 0) remains as slice index (depth=1 spacing)
    coords_pred[:, 1] /= height
    coords_pred[:, 2] /= width

    coords_true[:, 1] /= height
    coords_true[:, 2] /= width

    # Calculate Directed Hausdorff Distance (A->B and B->A)
    d_pred_true = directed_hausdorff(coords_pred, coords_true)[0]
    d_true_pred = directed_hausdorff(coords_true, coords_pred)[0]

    # Hausdorff distance is the maximum of the directed distances
    return max(d_pred_true, d_true_pred)

import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label, find_objects

from library.config import Config


def set_seed(seed=42):
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


def rle_encode(img):
    """
    Encodes a binary mask into RLE format (Run-Length Encoding).
    The competition format implies column-major ordering (top-to-bottom, then left-to-right).

    Args:
        img (np.array): Binary mask array (0s and 1s).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Transpose to handle column-major order (flatten goes row by row)
    pixels = img.T.flatten()
    # Pad with zeros to detect changes at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.array: Binary mask with the specified shape.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran order (column-major) to match encoding
    return img.reshape(shape, order="F")


def load_image(path):
    """
    Loads an image from the specified path, preserving bit depth (e.g., 16-bit).

    Args:
        path (str): Path to the image file.

    Returns:
        np.array: Image array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    # Load unchanged to preserve 16-bit depth
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # Ensure image has channel dimension if 2D
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def calculate_dice(y_true, y_pred):
    """
    Calculates the Dice coefficient.
    Formula: 2 * |X inter Y| / (|X| + |Y|)

    Args:
        y_true (np.array): Ground truth binary mask.
        y_pred (np.array): Predicted binary mask.

    Returns:
        float: Dice coefficient. Returns 0.0 if both masks are empty (per instructions).
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    im_sum = y_true.sum() + y_pred.sum()

    # Prompt specific: "Dice coefficient is defined to be 0 when both X and Y are empty."
    if im_sum == 0:
        return 0.0

    intersection = np.logical_and(y_true, y_pred).sum()
    return 2.0 * intersection / im_sum


def calculate_hausdorff(y_true_3d, y_pred_3d):
    """
    Calculates the 3D Hausdorff distance.
    Coordinates (y, x) are normalized by image dimensions. Z is treated with spacing 1.

    Args:
        y_true_3d (np.array): 3D Ground truth mask (Depth, Height, Width).
        y_pred_3d (np.array): 3D Predicted mask (Depth, Height, Width).

    Returns:
        float: Symmetric Hausdorff distance.
    """
    y_true_3d = np.asarray(y_true_3d).astype(bool)
    y_pred_3d = np.asarray(y_pred_3d).astype(bool)

    # Extract coordinates of non-zero pixels (z, y, x)
    true_points = np.argwhere(y_true_3d)
    pred_points = np.argwhere(y_pred_3d)

    # Handle empty cases
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        # If one is empty, return a large penalty or heuristic max distance
        # Since exact penalty isn't specified, we return 1.0 assuming normalized space dominance
        # or simply a high value. Given "bounded 0-1 score" context, 1.0 is a reasonable penalty.
        return 1.0

    depth, height, width = y_true_3d.shape

    # Convert to float for processing
    true_points = true_points.astype(float)
    pred_points = pred_points.astype(float)

    # Normalize Y (dim 1) and X (dim 2) by image dimensions
    # Z (dim 0) is left as is (spacing = 1)
    true_points[:, 1] /= height
    true_points[:, 2] /= width
    pred_points[:, 1] /= height
    pred_points[:, 2] /= width

    # Calculate directed Hausdorff distances
    d_ab = directed_hausdorff(true_points, pred_points)[0]
    d_ba = directed_hausdorff(pred_points, true_points)[0]

    # Symmetric Hausdorff distance
    return max(d_ab, d_ba)


def remove_small_objects_3d(mask_3d, min_size=10):
    """
    Removes small connected components from a 3D binary volume.

    Args:
        mask_3d (np.array): 3D binary mask (Depth, Height, Width).
        min_size (int): Minimum volume (in voxels) to keep a component.

    Returns:
        np.array: Cleaned 3D mask.
    """
    # Label connected components
    labeled_mask, num_features = label(mask_3d)

    if num_features == 0:
        return mask_3d

    # Find bounding boxes of objects
    objects = find_objects(labeled_mask)

    cleaned_mask = np.zeros_like(mask_3d)

    for i, slice_obj in enumerate(objects):
        if slice_obj is None:
            continue

        # Create a mask for the specific component
        # (i+1) because label 0 is background
        component_mask = labeled_mask[slice_obj] == (i + 1)

        if component_mask.sum() >= min_size:
            # Copy valid component to cleaned mask
            cleaned_mask[slice_obj][component_mask] = 1

    return cleaned_mask

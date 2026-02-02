import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from library.config import Config


def load_image(path):
    """
    Loads an image from the specified path.
    Handles 16-bit PNGs by reading unchanged.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image.
    """
    # Ensure path is absolute or relative to CWD correctly
    # If path is relative and not found, try joining with INPUT_DIR
    if not os.path.exists(path):
        alt_path = os.path.join(Config.INPUT_DIR, path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Image not found at {path}")

    # Read image with original depth (e.g. 16-bit)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image from {path}")

    return img


def rle_encode(img):
    """
    Run-length encoding for a binary mask.

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start length).
    """
    if img is None:
        return ""

    # Flatten column-wise (Fortran-style) as per competition format
    # Top-to-bottom, then left-to-right
    pixels = img.flatten(order="F")

    # We prepend and append 0 to detect runs at the start and end efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array now contains start indices of value changes.
    # Even indices (0, 2, ...) are starts of 1s (since we prepended 0)
    # Odd indices (1, 3, ...) are ends of 1s (starts of 0s)

    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a run-length encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # RLE is 1-based index, convert to 0-based
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise
    return img.reshape(shape, order="F")


def keep_largest_component_3d(segmentation_volume):
    """
    Post-processing to keep only the largest connected component in a 3D volume.
    This helps reduce Hausdorff distance by removing small floating artifacts.

    Args:
        segmentation_volume (np.ndarray): 3D binary array (Depth, Height, Width).

    Returns:
        np.ndarray: 3D binary array with only the largest component.
    """
    # Label connected components
    labeled_vol, num_features = ndimage.label(segmentation_volume)

    if num_features == 0:
        return segmentation_volume

    # Count size of each component
    # label 0 is background, so we slice from 1 to num_features
    component_sizes = ndimage.sum(
        segmentation_volume, labeled_vol, range(1, num_features + 1)
    )

    # Identify the label of the largest component
    # argmax returns index in the provided list, so we add 1 to get the label ID
    largest_component_label = np.argmax(component_sizes) + 1

    # Create mask for the largest component
    output_volume = np.zeros_like(segmentation_volume)
    output_volume[labeled_vol == largest_component_label] = 1

    return output_volume


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient between two binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: Dice coefficient score.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    return (2.0 * intersection + smooth) / (union + smooth)

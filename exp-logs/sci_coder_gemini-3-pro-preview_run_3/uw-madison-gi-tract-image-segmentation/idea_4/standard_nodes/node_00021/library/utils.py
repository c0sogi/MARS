import numpy as np
import cv2
import torch
import os
from scipy.ndimage import label
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.
    Wraps the library configuration's seeding function.
    """
    seed_everything(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    The competition format specifies that pixels are numbered from top to bottom,
    then left to right (Column-Major / Fortran order).

    Args:
        img (np.ndarray): Binary mask (0s and 1s), shape (H, W).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-wise (Fortran style) to match competition format
    pixels = img.flatten(order="F")

    # Prepend and append 0 to detect start/end of runs efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    # The length of a run is end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if str(mask_rle) == "nan" or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise (Fortran style) to match encoding
    return img.reshape(shape, order="F")


def keep_largest_component_3d(segmentation):
    """
    Performs 3D Connected Component Analysis (CCA) to retain only the largest
    connected component for a given class volume. This is critical for
    minimizing the Hausdorff distance metric by removing scattered outliers.

    Args:
        segmentation (np.ndarray): 3D binary mask, shape (D, H, W).

    Returns:
        np.ndarray: Filtered 3D binary mask containing only the largest object.
    """
    # Ensure input is binary
    segmentation = (segmentation > 0).astype(np.uint8)

    # Define 3D connectivity (26-connectivity: 3x3x3 cube of ones)
    structure = np.ones((3, 3, 3), dtype=np.int8)

    # Label connected components
    labeled_array, num_features = label(segmentation, structure=structure)

    if num_features == 0:
        return segmentation

    # Count pixels per label
    # bincount requires a 1D array of non-negative integers
    counts = np.bincount(labeled_array.ravel())

    # Ignore background (label 0)
    counts[0] = 0

    # If all non-background counts are 0
    if counts.max() == 0:
        return np.zeros_like(segmentation)

    # Identify the label with the largest pixel count
    max_label = counts.argmax()

    # Create mask for the largest component
    cleaned_segmentation = (labeled_array == max_label).astype(np.uint8)

    return cleaned_segmentation

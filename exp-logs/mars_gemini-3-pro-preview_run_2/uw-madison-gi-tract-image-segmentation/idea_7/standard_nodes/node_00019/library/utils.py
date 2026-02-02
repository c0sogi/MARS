import numpy as np
import pandas as pd
import cv2
import os
from scipy.ndimage import label
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    The pixels are numbered from top to bottom, then left to right (Fortran-style flattening).

    Args:
        img (np.ndarray): Binary mask (0s and 1s). Can be 2D or 3D, but will be flattened.

    Returns:
        str: Space-delimited string of 'start length start length ...'.
    """
    # Flatten in column-major order (Fortran-style) as per competition spec
    pixels = img.flatten(order="F")

    # We prepend and append 0 to detect transitions at the start and end of the array
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the start indices (1-based)
    # runs[1::2] are the end indices
    # Calculate lengths: end - start
    runs[1::2] -= runs[0::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape of the mask (height, width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill runs
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to 2D using Fortran-style (column-major) order
    return img.reshape(shape, order="F")


def keep_largest_component_3d(volume):
    """
    Performs 3D Connected Component Analysis on a binary volume and retains only the largest object.
    This is used to reduce false positives and improve the Hausdorff distance metric.

    Args:
        volume (np.ndarray): 3D binary volume of shape (Depth, Height, Width).

    Returns:
        np.ndarray: 3D binary volume containing only the largest connected component.
    """
    # Ensure volume is binary uint8
    volume_bin = (volume > 0).astype(np.uint8)

    # Label connected components in 3D
    # Default structure is None, which generates a 3x3x3 connectivity (26-connectivity)
    labeled_array, num_features = label(volume_bin)

    # If no features found, return empty volume
    if num_features == 0:
        return volume_bin

    # Count number of pixels in each component
    # bincount is efficient for this; ravel() avoids copying if memory layout permits
    counts = np.bincount(labeled_array.ravel())

    # Set background count (label 0) to 0 so it's not selected as the largest component
    counts[0] = 0

    # Identify the label with the maximum pixel count
    max_label = counts.argmax()

    # Return mask corresponding to the largest component
    return (labeled_array == max_label).astype(np.uint8)

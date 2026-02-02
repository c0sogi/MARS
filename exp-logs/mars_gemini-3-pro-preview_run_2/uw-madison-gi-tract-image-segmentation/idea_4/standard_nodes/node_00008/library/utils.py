import os
import cv2
import numpy as np
import pandas as pd
from library import config


def load_image(path, resize_to=None):
    """
    Loads an image from the given path.
    Handles 16-bit PNGs by normalizing to 0-1 range.
    Resizes the image to the target size if specified.

    Args:
        path (str): Path to the image file.
        resize_to (tuple): Target size (height, width). Defaults to config.IMG_SIZE.

    Returns:
        np.ndarray: Normalized image array (float32).
    """
    # Construct full path if relative path is provided
    if not os.path.isabs(path):
        # Try joining with INPUT_DIR first
        full_path = os.path.join(config.INPUT_DIR, path)
        # If that doesn't exist, check if it's relative to CWD
        if not os.path.exists(full_path) and os.path.exists(path):
            full_path = path
    else:
        full_path = path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image file not found: {full_path}")

    # Read image
    # IMREAD_UNCHANGED is crucial for 16-bit images
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {full_path}")

    if resize_to is None:
        resize_to = config.IMG_SIZE

    # Normalize based on bit depth
    if img.dtype == np.uint16:
        # 16-bit image
        img = img.astype(np.float32) / 65535.0
    else:
        # Assume 8-bit
        img = img.astype(np.float32) / 255.0

    # Resize if necessary
    if resize_to is not None:
        h, w = img.shape[:2]
        if h != resize_to[0] or w != resize_to[1]:
            img = cv2.resize(
                img, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA
            )

    return img


def rle_encode(mask):
    """
    Run-length encodes a binary mask.
    Pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        str: Space-separated RLE string.
    """
    # Flatten in column-major order (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Check if empty
    if np.sum(pixels) == 0:
        return ""

    # Pad with zeros to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    # pixels[1:] != pixels[:-1] gives boolean array
    # np.where gives indices into that array
    # +1 adjusts to 1-based indexing for the RLE format
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end - start
    # runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a run-length encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1

    # Calculate end indices
    ends = starts + lengths

    # Create flat array
    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image (column-major)
    return img.reshape(shape, order="F")


def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary masks.
    Formula: 2 * |X n Y| / (|X| + |Y|)
    Returns 0 if both masks are empty (per task spec).

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.

    Returns:
        float: Dice coefficient.
    """
    # Flatten to ensure 1D
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    sum_true = np.sum(y_true_f)
    sum_pred = np.sum(y_pred_f)

    denominator = sum_true + sum_pred

    if denominator == 0:
        return 0.0

    return (2.0 * intersection) / denominator


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    if split == "train":
        path = os.path.join(config.METADATA_DIR, "train_metadata.csv")
    elif split == "val":
        path = os.path.join(config.METADATA_DIR, "val_metadata.csv")
    elif split == "test":
        path = os.path.join(config.METADATA_DIR, "test_metadata.csv")
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)

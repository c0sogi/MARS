import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from library.config import Config


def set_seed(seed=42):
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
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (Height, Width).
                          0 - background, 1 - foreground.

    Returns:
        str: Space-delimited RLE string.
    """
    # The competition specifies pixels are numbered from top to bottom, then left to right.
    # This corresponds to Fortran-style flattening (column-major).
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask of shape (Height, Width).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran-style to match the encoding direction
    return img.reshape(shape, order="F")


def compute_dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Computes the Dice Coefficient.

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

    # Flatten
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def load_and_preprocess_metadata(csv_path, load_cached_data=True):
    """
    Loads metadata from CSV, preprocesses it (sorting, type conversion),
    and caches the result to Parquet for faster subsequent access.

    Args:
        csv_path (str): Path to the source CSV file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Preprocessed metadata DataFrame.
    """
    # Define cache path
    filename = os.path.basename(csv_path).replace(".csv", ".parquet")
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reloading from CSV.")

    # 2. Process from scratch
    df = pd.read_csv(csv_path, keep_default_na=False)

    # Ensure slice is integer for correct sorting
    if "slice" in df.columns:
        # Extract numeric part if it's a string, though usually it's clean in metadata
        df["slice_idx"] = df["slice"].astype(str).str.extract(r"(\d+)").astype(int)

    # Sort by Case, Day, then Slice index to ensure 3D consistency
    sort_cols = [c for c in ["case", "day", "slice_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def group_metadata_by_case(df):
    """
    Groups the metadata DataFrame by (case, day) to facilitate 3D processing.

    Args:
        df (pd.DataFrame): Metadata DataFrame.

    Returns:
        dict: Dictionary where keys are (case, day) tuples and values are DataFrames
              containing all slices for that scan, sorted by slice index.
    """
    grouped = {}
    # Ensure we have the sorting column
    if "slice_idx" not in df.columns:
        df["slice_idx"] = df["slice"].astype(str).str.extract(r"(\d+)").astype(int)

    for (case, day), group in df.groupby(["case", "day"]):
        # Ensure sorted by slice
        grouped[(case, day)] = group.sort_values("slice_idx").reset_index(drop=True)

    return grouped


def post_process_largest_component(mask):
    """
    Post-processing to keep only the largest connected component for a 2D mask.
    This helps remove noise.

    Args:
        mask (np.ndarray): Binary mask (H, W).

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    # Ensure uint8
    mask = mask.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    # stats[:, 4] is the area. Label 0 is background.
    # Find label with max area excluding background
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == max_label] = 1

    return cleaned_mask

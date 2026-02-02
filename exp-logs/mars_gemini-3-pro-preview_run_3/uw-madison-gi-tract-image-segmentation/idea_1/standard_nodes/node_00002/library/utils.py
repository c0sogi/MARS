import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W), where 1 indicates mask.

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten column-wise (Fortran-style) as per competition usually,
    # but the prompt says "top to bottom, then left to right".
    # Pixel 1 is (1,1), Pixel 2 is (2,1). This corresponds to flattening column-major (F).
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
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

    # Reshape column-major to match the encoding direction
    return img.reshape(shape, order="F")


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Calculates the Dice coefficient for a batch of predictions.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary masks.
        y_true (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Mean Dice coefficient.
    """
    # Flatten tensors
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    return (2.0 * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)


def get_metadata(load_cached_data=True):
    """
    Loads metadata for train, validation, and test sets.
    Implements caching mechanism using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_metadata.parquet")
    val_cache_path = os.path.join(cache_dir, "val_metadata.parquet")
    test_cache_path = os.path.join(cache_dir, "test_metadata.parquet")

    # Check if cache exists and loading is requested
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    ):

        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        return train_df, val_df, test_df

    # Process from scratch
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Fill NaNs in segmentation columns
    mask_cols = ["large_bowel", "small_bowel", "stomach"]
    for col in mask_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("")
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna("")

    # Save to cache
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, val_df, test_df


def compute_hausdorff_3d(
    pred_vol, true_vol, spacing_h=1.5, spacing_w=1.5, slice_depth=3.0
):
    """
    Approximates 3D Hausdorff distance between two binary volumes.

    Args:
        pred_vol (np.ndarray): Predicted binary volume (D, H, W).
        true_vol (np.ndarray): Ground truth binary volume (D, H, W).
        spacing_h (float): Pixel spacing in height (mm).
        spacing_w (float): Pixel spacing in width (mm).
        slice_depth (float): Slice thickness (mm).

    Returns:
        float: The directed Hausdorff distance.
    """
    from scipy.spatial.distance import directed_hausdorff

    # Get coordinates of non-zero pixels
    pred_coords = np.argwhere(pred_vol > 0)
    true_coords = np.argwhere(true_vol > 0)

    if len(pred_coords) == 0 or len(true_coords) == 0:
        # If either is empty, distance is undefined or max.
        # For this competition, usually handled by returning a penalty or 0 if both empty.
        if len(pred_coords) == 0 and len(true_coords) == 0:
            return 0.0
        return 1.0  # Normalized penalty, though real HD would be large.

    # Scale coordinates by physical spacing
    # argwhere returns (z, y, x) -> (slice, height, width)
    scale = np.array([slice_depth, spacing_h, spacing_w])

    pred_points = pred_coords * scale
    true_points = true_coords * scale

    # Calculate directed Hausdorff distance both ways and take max
    d_forward = directed_hausdorff(pred_points, true_points)[0]
    d_backward = directed_hausdorff(true_points, pred_points)[0]

    return max(d_forward, d_backward)

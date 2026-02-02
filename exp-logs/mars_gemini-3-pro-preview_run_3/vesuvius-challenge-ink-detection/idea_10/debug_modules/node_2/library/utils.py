import os
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from library.config import Config


def get_boundary_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Generates the boundary mask for auxiliary supervision.

    Calculates the morphological difference between the dilated and eroded
    versions of the binary mask to isolate edges.

    Args:
        mask: Binary numpy array (0 and 1).
        kernel_size: Size of the structuring element.

    Returns:
        Binary boundary mask (0 and 1).
    """
    # Ensure mask is uint8
    mask_uint8 = (
        (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)

    # Difference captures the boundary
    boundary = cv2.absdiff(dilated, eroded)

    # Binarize back to 0/1
    boundary = (boundary > 127).astype(np.float32)
    return boundary


def rle_encode(img: np.ndarray) -> str:
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The pixels are numbered from left to right, then top to bottom.

    Args:
        img: Binary mask (2D numpy array).

    Returns:
        Space-delimited string of start positions and run lengths.
    """
    pixels = img.flatten()
    # We add 0 at the beginning and end to detect transitions at edges
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run (exclusive)
    # The length is runs[1] - runs[0]
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def f05_score(preds, labels, threshold: float = 0.5, epsilon: float = 1e-7):
    """
    Calculates the F0.5 score.

    Supports both numpy arrays and torch tensors.

    Args:
        preds: Predicted probabilities or binary mask.
        labels: Ground truth binary mask.
        threshold: Binarization threshold for probabilities.
        epsilon: Small constant to prevent division by zero.

    Returns:
        F0.5 score (float).
    """
    beta = 0.5

    # Handle Tensor inputs
    if torch.is_tensor(preds):
        preds = (preds > threshold).float()
        labels = labels.float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()

        score = ((1 + beta**2) * tp + epsilon) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + epsilon
        )
        return score.item()

    # Handle Numpy inputs
    else:
        preds_bin = (preds > threshold).astype(np.float32)
        labels = labels.astype(np.float32)

        tp = np.sum(preds_bin * labels)
        fp = np.sum(preds_bin * (1 - labels))
        fn = np.sum((1 - preds_bin) * labels)

        score = ((1 + beta**2) * tp + epsilon) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + epsilon
        )
        return float(score)


def load_normalization_stats(metadata_path: Path, load_cached_data: bool = True):
    """
    Computes or loads global mean and standard deviation for the dataset.

    Uses an incremental algorithm to avoid loading all volumes into memory.
    Caches the result in the working directory.

    Args:
        metadata_path: Path to the training metadata CSV.
        load_cached_data: If True, attempts to load from cache first.

    Returns:
        Tuple (mean, std).
    """
    cache_path = Config.WORKING_DIR / "normalization_stats.npy"

    # 1. Try to load from cache
    if load_cached_data and cache_path.exists():
        try:
            stats = np.load(cache_path)
            # print(f"Loaded normalization stats from cache: Mean={stats[0]:.4f}, Std={stats[1]:.4f}")
            return stats[0], stats[1]
        except Exception:
            pass  # Fallback to recomputing

    # 2. Compute from scratch
    # print("Computing normalization stats from training volumes...")
    df = pd.read_csv(metadata_path)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for _, row in df.iterrows():
        vol_path = Config.INPUT_DIR / row["surface_volume_path"]

        # Iterate over all slices in the volume
        # We assume slices are named 00.tif, 01.tif, etc.
        # Config.Z_DIM defines the depth
        for z in range(Config.Z_DIM):
            slice_path = vol_path / f"{z:02d}.tif"
            if not slice_path.exists():
                continue

            img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Accumulate stats
            img_flat = img.astype(np.float64)
            total_sum += np.sum(img_flat)
            total_sq_sum += np.sum(img_flat**2)
            total_count += img_flat.size

    if total_count == 0:
        # Fallback if no data found
        return 0.0, 1.0

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean**2)
    std = np.sqrt(variance)

    # 3. Save to cache
    np.save(cache_path, np.array([mean, std]))
    # print(f"Computed and saved normalization stats: Mean={mean:.4f}, Std={std:.4f}")

    return mean, std

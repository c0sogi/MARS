import os
import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Run-length encoding for the submission.
    Args:
        mask (np.array): Binary mask (0 or 1) of shape (H, W).
    Returns:
        str: Space-delimited run-length encoding.
    """
    # Flatten the mask (row-major order: left to right, then top to bottom)
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, smooth=1e-6, threshold=None):
    """
    Calculates the F-beta score.
    Args:
        preds (torch.Tensor or np.array): Predictions (probabilities or binary).
        targets (torch.Tensor or np.array): Ground truth labels.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        smooth (float): Smoothing factor to avoid division by zero.
        threshold (float, optional): If provided, binarizes preds before calculation.
    Returns:
        float: The F-beta score.
    """
    if torch.is_tensor(preds):
        if threshold is not None:
            preds = (preds > threshold).float()

        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

    else:
        # Numpy implementation
        if threshold is not None:
            preds = (preds > threshold).astype(float)

        preds = preds.flatten()
        targets = targets.flatten()

        tp = np.sum(preds * targets)
        fp = np.sum(preds * (1 - targets))
        fn = np.sum((1 - preds) * targets)

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)

    if torch.is_tensor(score):
        return score.item()
    return float(score)


def _get_fragment_row(fragment_id, split, metadata_df):
    """Helper to retrieve the metadata row for a specific fragment."""
    # Ensure fragment_id types match (metadata might be int or str)
    row = metadata_df[metadata_df["fragment_id"].astype(str) == str(fragment_id)]
    if row.empty:
        raise ValueError(f"Fragment ID {fragment_id} not found in {split} metadata.")
    return row.iloc[0]


def load_volume(fragment_id, split, metadata_df, load_cached_data=True):
    """
    Loads the 3D surface volume for a fragment.
    Args:
        fragment_id: ID of the fragment.
        split: 'train', 'val', or 'test'.
        metadata_df: DataFrame containing path information.
        load_cached_data: Whether to use caching.
    Returns:
        np.array: 3D volume of shape (Z, H, W).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = Config.WORKING_DIR / f"{split}_{fragment_id}_volume.npy"

    if load_cached_data and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(
                f"Failed to load cached volume for {fragment_id}: {e}. Recomputing..."
            )

    # Compute from scratch
    row = _get_fragment_row(fragment_id, split, metadata_df)
    volume_dir = Config.INPUT_DIR / row["surface_volume_path"]

    slices = []
    for i in range(Config.Z_DEPTH):
        filename = f"{i:02d}.tif"
        file_path = volume_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Slice {filename} missing at {volume_dir}")

        # Load grayscale
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")
        slices.append(img)

    # Stack to (Z, H, W)
    volume = np.stack(slices, axis=0)

    # Save to cache
    np.save(cache_path, volume)

    return volume


def load_mask(fragment_id, split, metadata_df, load_cached_data=True):
    """
    Loads the valid pixel mask for a fragment.
    Args:
        fragment_id: ID of the fragment.
        split: 'train', 'val', or 'test'.
        metadata_df: DataFrame containing path information.
        load_cached_data: Whether to use caching.
    Returns:
        np.array: Binary mask of shape (H, W).
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = Config.WORKING_DIR / f"{split}_{fragment_id}_mask.npy"

    if load_cached_data and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cached mask for {fragment_id}: {e}. Recomputing...")

    row = _get_fragment_row(fragment_id, split, metadata_df)
    mask_path = Config.INPUT_DIR / row["mask_path"]

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    # Binarize (0 or 1)
    mask = (mask > 0).astype(np.uint8)

    np.save(cache_path, mask)
    return mask


def load_inklabels(fragment_id, split, metadata_df, load_cached_data=True):
    """
    Loads the ink labels for a fragment.
    Args:
        fragment_id: ID of the fragment.
        split: 'train', 'val', or 'test'.
        metadata_df: DataFrame containing path information.
        load_cached_data: Whether to use caching.
    Returns:
        np.array: Binary labels of shape (H, W).
    """
    # Only train/val have labels
    if split == "test":
        return None

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = Config.WORKING_DIR / f"{split}_{fragment_id}_label.npy"

    if load_cached_data and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cached label for {fragment_id}: {e}. Recomputing...")

    row = _get_fragment_row(fragment_id, split, metadata_df)

    if pd.isna(row.get("inklabels_path")):
        return None

    label_path = Config.INPUT_DIR / row["inklabels_path"]

    if not label_path.exists():
        raise FileNotFoundError(f"Ink labels not found at {label_path}")

    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Failed to read ink labels: {label_path}")

    # Binarize
    label = (label > 0).astype(np.uint8)

    np.save(cache_path, label)
    return label

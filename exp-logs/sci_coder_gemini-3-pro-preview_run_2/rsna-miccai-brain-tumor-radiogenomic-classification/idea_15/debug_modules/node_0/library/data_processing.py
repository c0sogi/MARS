import os
import re
import numpy as np
import pandas as pd
import cv2
import torch
from typing import List, Dict, Optional, Tuple

from library.config import Config
from library.utils import get_logger
from library.dicom_utils import read_dicom_file

# Initialize logger
logger = get_logger("DataProcessing")


def natural_sort_key(s: str) -> List[int]:
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm, Image-2.dcm, Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_sorted_files(dir_path: str) -> List[str]:
    """
    Returns a list of filenames in the directory sorted by slice number.
    """
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    files.sort(key=natural_sort_key)
    return files


def calculate_roi_index(flair_path: str) -> int:
    """
    Calculates the anchor slice index using the Integral-ROI Pipeline.

    Strategy:
    1. Load all FLAIR slices.
    2. Compute sum of intensity (integral) for each slice.
    3. Apply Moving Average Filter.
    4. Select max intensity within 15%-85% depth bounds.

    Args:
        flair_path (str): Path to the FLAIR directory.

    Returns:
        int: The index of the anchor slice in the sorted file list.
             Returns middle index if processing fails.
    """
    files = get_sorted_files(flair_path)
    num_files = len(files)

    if num_files == 0:
        return 0

    if num_files < 5:
        return num_files // 2

    # 1. Calculate Intensity Profile
    intensities = []
    valid_indices = []

    # We read every slice to compute the profile.
    # This is heavy, hence why caching in get_roi_cache is essential.
    for i, f in enumerate(files):
        f_path = os.path.join(flair_path, f)
        try:
            img = read_dicom_file(f_path)
            # Sum of intensity
            val = np.sum(img)
            intensities.append(val)
            valid_indices.append(i)
        except Exception:
            intensities.append(0)
            valid_indices.append(i)

    intensities = np.array(intensities, dtype=np.float32)

    # 2. Apply Moving Average Filter (Smoothing)
    window = Config.ROI_SMOOTH_WINDOW
    if len(intensities) >= window:
        kernel = np.ones(window) / window
        # mode='same' returns output of same length as input
        smoothed = np.convolve(intensities, kernel, mode="same")
    else:
        smoothed = intensities

    # 3. Apply Depth Bounds (15% - 85%)
    min_depth = int(num_files * Config.ROI_DEPTH_MIN)
    max_depth = int(num_files * Config.ROI_DEPTH_MAX)

    # Ensure bounds are valid
    if min_depth >= max_depth:
        min_depth = 0
        max_depth = num_files

    # Slice the valid range
    roi_region = smoothed[min_depth:max_depth]

    if len(roi_region) == 0:
        return num_files // 2

    # Find argmax relative to the sliced region
    relative_argmax = np.argmax(roi_region)

    # Convert back to absolute index
    anchor_index = min_depth + relative_argmax

    return int(anchor_index)


def get_roi_cache(df: pd.DataFrame, load_cached_data: bool = True) -> Dict[str, int]:
    """
    Generates or loads a cache mapping BraTS21ID to the calculated ROI anchor index.

    Args:
        df (pd.DataFrame): DataFrame containing 'BraTS21ID' and 'path_FLAIR'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        Dict[str, int]: Dictionary mapping BraTS21ID (as string) to anchor index.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "roi_cache.parquet")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading ROI cache from {cache_path}")
            cache_df = pd.read_parquet(cache_path)
            # Convert to dict: BraTS21ID (str) -> anchor_index (int)
            # Ensure ID is string to match metadata conventions often used
            cache_map = pd.Series(
                cache_df.anchor_index.values, index=cache_df.BraTS21ID.astype(str)
            ).to_dict()

            # Verify coverage: Check if current df IDs are in cache
            current_ids = set(df["BraTS21ID"].astype(str))
            cached_ids = set(cache_map.keys())

            if current_ids.issubset(cached_ids):
                return cache_map
            else:
                logger.info("Cache incomplete. Recomputing missing entries...")
                # We will fall through to computation, but we can seed with existing
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")
            cache_map = {}
    else:
        cache_map = {}

    # 2. Compute Missing ROIs
    logger.info("Computing ROI indices (Integral-Statistic Pipeline)...")

    new_entries = []
    total = len(df)

    for idx, row in df.iterrows():
        subject_id = str(row["BraTS21ID"])

        # Skip if already in cache
        if subject_id in cache_map:
            continue

        flair_rel_path = row["path_FLAIR"]
        flair_full_path = os.path.join(Config.INPUT_DIR, flair_rel_path)

        if os.path.exists(flair_full_path):
            anchor_idx = calculate_roi_index(flair_full_path)
        else:
            anchor_idx = 0  # Fallback

        cache_map[subject_id] = anchor_idx
        new_entries.append({"BraTS21ID": subject_id, "anchor_index": anchor_idx})

        if len(new_entries) % 50 == 0:
            logger.info(f"Processed {len(new_entries)} new subjects...")

    # 3. Save Cache
    # Convert full map to dataframe for saving
    full_cache_df = pd.DataFrame(
        [{"BraTS21ID": k, "anchor_index": v} for k, v in cache_map.items()]
    )

    # Save as parquet
    try:
        full_cache_df.to_parquet(cache_path)
        logger.info(f"ROI cache saved to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save ROI cache: {e}")

    return cache_map


def process_image(img: np.ndarray) -> np.ndarray:
    """
    Applies standard preprocessing: Resize and Min-Max Normalize.
    """
    # Resize
    # Use INTER_AREA for shrinking to avoid aliasing artifacts
    img_resized = cv2.resize(img, Config.IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Min-Max Normalization to [0, 1]
    min_val = np.min(img_resized)
    max_val = np.max(img_resized)

    if max_val - min_val > 1e-6:
        img_norm = (img_resized - min_val) / (max_val - min_val)
    else:
        img_norm = np.zeros_like(img_resized)

    return img_norm


def load_patient_volume(row: pd.Series, anchor_index: int) -> torch.Tensor:
    """
    Loads the patient volume based on the anchor index and Modality-Isolated logic.

    Structure:
    - Modalities: FLAIR, T1w, T1wCE, T2w
    - Slices: Anchor-5, Anchor, Anchor+5 (Stride 5)
    - Total Channels: 4 * 3 = 12

    Args:
        row (pd.Series): Row from metadata dataframe containing paths.
        anchor_index (int): The calculated anchor slice index.

    Returns:
        torch.Tensor: Tensor of shape (12, 224, 224).
    """
    channels = []

    # Define relative offsets
    offsets = [-Config.ROI_STRIDE, 0, Config.ROI_STRIDE]

    for mod in Config.MODALITIES:
        path_col = f"path_{mod}"
        if path_col not in row:
            # Handle case where path might be missing (shouldn't happen with valid metadata)
            # Fill with zeros for this modality
            for _ in offsets:
                channels.append(np.zeros(Config.IMG_SIZE, dtype=np.float32))
            continue

        dir_path = os.path.join(Config.INPUT_DIR, row[path_col])
        files = get_sorted_files(dir_path)
        num_files = len(files)

        if num_files == 0:
            # Modality missing/empty
            for _ in offsets:
                channels.append(np.zeros(Config.IMG_SIZE, dtype=np.float32))
            continue

        # Extract slices
        for offset in offsets:
            target_idx = anchor_index + offset

            # Edge Clamping
            if target_idx < 0:
                target_idx = 0
            elif target_idx >= num_files:
                target_idx = num_files - 1

            file_name = files[target_idx]
            file_path = os.path.join(dir_path, file_name)

            # Read
            img = read_dicom_file(file_path)

            # Process (Resize + Normalize)
            img_proc = process_image(img)

            channels.append(img_proc)

    # Stack channels
    # Result shape: (12, 224, 224)
    volume = np.stack(channels, axis=0)

    # Convert to Tensor
    return torch.tensor(volume, dtype=torch.float32)

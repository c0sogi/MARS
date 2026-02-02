import os
import re
import numpy as np
import pandas as pd
from library.config import Config
from library.dicom_processing import read_dicom_robust


def extract_number(filename):
    """
    Extracts the integer number from a filename like 'Image-123.dcm'.
    Used for sorting DICOM files numerically.
    """
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def get_best_slice_index(dir_path, method):
    """
    Determines the best slice index within the configured depth range
    based on the specified method (sum or max intensity).

    Args:
        dir_path (str): Path to the modality directory.
        method (str): 'sum' for integral intensity, 'max' for peak intensity.

    Returns:
        int: The 0-based index of the selected slice within the sorted file list.
    """
    if not os.path.exists(dir_path):
        return 0

    # List and sort files numerically to ensure correct volumetric order
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    if not files:
        return 0

    files.sort(key=extract_number)

    num_files = len(files)

    # Define search bounds based on Config to avoid noise at volume ends
    start_idx = int(num_files * Config.DEPTH_MIN)
    end_idx = int(num_files * Config.DEPTH_MAX)

    # Handle cases where the range is invalid (e.g., very few slices)
    if start_idx >= end_idx:
        # Fallback to middle slice
        return num_files // 2

    best_score = -1.0
    best_idx = start_idx

    # Iterate through the valid range
    for i in range(start_idx, end_idx):
        file_path = os.path.join(dir_path, files[i])

        # Use robust reader to get image data (uint16)
        img = read_dicom_robust(file_path)

        # Calculate score based on method
        if method == "sum":
            # Use float64 accumulator to prevent overflow
            score = np.sum(img, dtype=np.float64)
        elif method == "max":
            score = np.max(img)
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def compute_roi_indices(metadata_df):
    """
    Iterates through the metadata dataframe and computes ROI indices
    using the single anchor defined in Config.
    """
    results = []

    # Iterate through subjects
    for idx, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]

        # --- Anchor Processing (e.g., FLAIR Sum) ---
        path_col = f"path_{Config.ANCHOR_MODALITY}"
        if path_col in row:
            dir_path = os.path.join(Config.INPUT_DIR, row[path_col])
            idx_roi = get_best_slice_index(dir_path, Config.ANCHOR_METHOD)
        else:
            idx_roi = 0

        results.append(
            {
                "BraTS21ID": subject_id,
                "roi_idx": idx_roi,
            }
        )

    return pd.DataFrame(results)


def get_roi_indices(metadata_df, split_name="train", load_cached_data=True):
    """
    Main entry point to get ROI indices. Handles caching logic using Parquet.

    Args:
        metadata_df (pd.DataFrame): Metadata containing subject IDs and paths.
        split_name (str): Name of the split (train/val/test) for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with BraTS21ID and roi_idx.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"roi_indices_{split_name}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached ROI indices for {split_name} from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch if cache miss or force reload
    print(f"Computing ROI indices for {split_name} (Single-Anchor Strategy)...")
    df_indices = compute_roi_indices(metadata_df)

    # 3. Save to cache
    print(f"Saving ROI indices to {cache_path}")
    df_indices.to_parquet(cache_path, index=False)

    return df_indices

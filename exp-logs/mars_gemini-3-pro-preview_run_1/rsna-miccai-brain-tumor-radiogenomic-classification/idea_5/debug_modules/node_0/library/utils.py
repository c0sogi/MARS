import os
import random
import re
import numpy as np
import torch
import cv2
import pydicom
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_number(filename):
    """
    Helper to extract the integer number from a filename like 'Image-10.dcm'
    for correct numerical sorting.
    """
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def get_sorted_dicom_files(folder_path):
    """
    Returns a sorted list of DICOM filenames in a directory.
    """
    if not os.path.exists(folder_path):
        return []
    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    files.sort(key=_extract_number)
    return files


def read_dicom(path):
    """
    Reads a DICOM file and returns the pixel array.
    Uses pydicom as the primary reader.
    """
    try:
        ds = pydicom.dcmread(path)
        return ds.pixel_array
    except Exception:
        # Fallback to returning an empty array which will be handled downstream
        # or try opencv if pydicom fails on specific compression
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except Exception:
            pass
        return None


def load_processed_slice(path, target_size=Config.IMG_SIZE):
    """
    Loads a DICOM slice, resizes it, converts to float32, and applies
    instance-level min-max normalization.

    Args:
        path (str): Full path to the DICOM file.
        target_size (int): Desired output spatial dimension (square).

    Returns:
        np.ndarray: Processed image of shape (target_size, target_size) with values in [0, 1].
    """
    # Initialize empty canvas
    if not path or not os.path.exists(path):
        return np.zeros((target_size, target_size), dtype=np.float32)

    img = read_dicom(path)

    if img is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Resize
    # CV2 expects (W, H), but since it's square it doesn't matter
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Convert to float32 to preserve dynamic range
    img = img.astype(np.float32)

    # Instance-level Min-Max Normalization
    # We avoid ImageNet mean/std as medical images have different distributions
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = img - min_val  # Effectively zeroing if flat

    return img


def generate_slice_cache(metadata_df, split_name, load_cached_data=True):
    """
    Deterministically selects file paths for the required depths (e.g., 45%, 50%, 55%)
    for all subjects in the metadata DataFrame and caches the result to Parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing subject IDs and modality paths.
        split_name (str): 'train', 'val', or 'test' to identify the cache file.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: DataFrame with columns for each modality-depth combination containing file paths.
    """
    cache_file = os.path.join(
        Config.CACHE_DIR, f"cached_file_lists_{split_name}.parquet"
    )
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached slice paths from {cache_file}")
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache ({e}), recomputing...")

    print(f"Generating slice cache for {split_name}...")

    results = []

    # Iterate over all subjects
    for _, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]
        record = {"BraTS21ID": sid}

        # Preserve label if it exists
        if "MGMT_value" in row:
            record["MGMT_value"] = row["MGMT_value"]

        # For each modality (FLAIR, T1wCE, T2w)
        for mod in Config.SELECTED_MODALITIES:
            # Map Config modality name (e.g., 'FLAIR') to metadata column (e.g., 'flair_path')
            col_name = f"{mod.lower()}_path"
            rel_path = row[col_name]
            full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Get all sorted files
            files = get_sorted_dicom_files(full_dir_path)
            n_files = len(files)

            # Select specific slices based on depth percentages
            for depth in Config.SLICE_DEPTHS:
                if n_files == 0:
                    file_path = ""
                else:
                    # Calculate index
                    idx = int(n_files * depth)
                    # Clamp index
                    idx = min(max(0, idx), n_files - 1)
                    file_path = os.path.join(full_dir_path, files[idx])

                # Store in record with key like 'FLAIR_0.45'
                key = f"{mod}_{depth}"
                record[key] = file_path

        results.append(record)

    # 2. Save result to cache
    df_cache = pd.DataFrame(results)
    df_cache.to_parquet(cache_file)
    print(f"Saved slice cache to {cache_file}")

    return df_cache

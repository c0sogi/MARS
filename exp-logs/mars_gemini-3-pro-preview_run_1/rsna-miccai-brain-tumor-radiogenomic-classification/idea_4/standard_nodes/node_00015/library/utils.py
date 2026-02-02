import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import torch
import random
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the Config.set_seed method.
    """
    Config.set_seed(seed)


def read_dicom(path, size=Config.IMG_SIZE):
    """
    Reads a DICOM file, normalizes pixel values to 0-255, and resizes the image.

    Args:
        path (str): Path to the .dcm file.
        size (int): Target spatial dimension (size x size).

    Returns:
        np.ndarray: Processed image array of shape (size, size) with dtype uint8.
    """
    img = None

    # Attempt 1: Try using pydicom (standard for DICOM)
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
    except Exception:
        pass

    # Attempt 2: Fallback to OpenCV if pydicom fails or is not installed
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Handle read failure by returning a black image
    if img is None:
        return np.zeros((size, size), dtype=np.uint8)

    # Handle potential 3D output from single-frame DICOMs (rare but possible)
    if img.ndim == 3:
        img = img[0]

    # Normalize to 0-255 range
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min  # Results in array of zeros

    img = (img * 255).astype(np.uint8)

    # Resize to target dimension
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


def get_sorted_file_list(folder_path):
    """
    Retrieves a sorted list of DICOM files from a directory.
    Files are sorted numerically based on the index in the filename (e.g., Image-10.dcm).

    Args:
        folder_path (str): Path to the directory containing DICOM files.

    Returns:
        list: Sorted list of full file paths.
    """
    if not os.path.exists(folder_path):
        return []

    files = glob.glob(os.path.join(folder_path, "*.dcm"))

    def extract_number(f):
        # Extracts the number N from 'Image-N.dcm'
        basename = os.path.basename(f)
        match = re.search(r"Image-(\d+)\.dcm", basename)
        if match:
            return int(match.group(1))
        return 0

    return sorted(files, key=extract_number)


def get_all_file_lists(df, load_cached_data=True, split_name="train"):
    """
    Generates a DataFrame containing sorted file lists for all subjects in the input DataFrame.
    Implements caching to Parquet to avoid redundant filesystem scanning.

    Args:
        df (pd.DataFrame): Metadata DataFrame containing 'BraTS21ID' and modality paths.
        load_cached_data (bool): Whether to attempt loading from cache.
        split_name (str): Identifier for the split (train/val/test) to name the cache file.

    Returns:
        pd.DataFrame: DataFrame with columns 'BraTS21ID' and '{modality}_files' (list of paths).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cached_file_lists_{split_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            if not cached_df.empty:
                return cached_df
        except Exception:
            # Proceed to compute if cache load fails
            pass

    # 2. Compute file lists from scratch
    data = []
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for _, row in df.iterrows():
        sid = row["BraTS21ID"]
        record = {"BraTS21ID": sid}

        for mod in modalities:
            # Metadata columns are lowercase (e.g., 'flair_path')
            col_name = f"{mod.lower()}_path"
            if col_name in row:
                rel_path = row[col_name]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)
                # Get numerically sorted list of files
                files = get_sorted_file_list(full_path)
                record[f"{mod}_files"] = files
            else:
                record[f"{mod}_files"] = []

        data.append(record)

    result_df = pd.DataFrame(data)

    # 3. Save to cache
    try:
        result_df.to_parquet(cache_path)
    except Exception:
        pass

    return result_df

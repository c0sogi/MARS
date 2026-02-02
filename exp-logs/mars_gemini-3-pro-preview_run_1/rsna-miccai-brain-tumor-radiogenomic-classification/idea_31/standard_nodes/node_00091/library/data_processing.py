import os
import numpy as np
import pandas as pd
import cv2
from library.config import load_dicom_image, WORKING_DIR, INPUT_DIR

# Ensure working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)


def load_dicom_slice(path):
    """
    Reads a DICOM file using the library utility and converts it to float32.

    Args:
        path (str): Full path to the DICOM file.

    Returns:
        np.ndarray or None: The image array in float32, or None if loading fails.
    """
    img = load_dicom_image(path)
    if img is None:
        return None
    return img.astype(np.float32)


def normalize_minmax(img):
    """
    Scales the image to [0, 1] using min-max normalization.
    Handles cases where the image is constant (max == min) by returning zeros.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image.
    """
    if img is None:
        return None

    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)


def get_relative_indices(start, end, total_count, depths):
    """
    Calculates the integer slice indices corresponding to relative depths
    within the ROI (start to end).

    Args:
        start (int): ROI start index.
        end (int): ROI end index.
        total_count (int): Total number of files in the directory.
        depths (list of float): Relative depths (e.g., [0.4, 0.5, 0.6]).

    Returns:
        list of int: Calculated zero-based indices.
    """
    roi_len = end - start
    indices = []

    for d in depths:
        # If ROI is invalid or flat, default to start
        if roi_len < 1:
            idx = start
        else:
            idx = int(start + roi_len * d)

        # Clamp to valid range [0, total_count - 1]
        idx = max(0, min(idx, total_count - 1))
        indices.append(idx)

    return indices


def get_modality_roi(directory_path):
    """
    Scans a directory of DICOM files to find the start and end indices
    of the brain tissue (pixels > 0).

    Args:
        directory_path (str): Path to the directory containing DICOM files.

    Returns:
        tuple: (start_index, end_index, total_count)
    """
    if not os.path.exists(directory_path):
        return 0, 0, 0

    # Sort files numerically by instance number (Image-N.dcm)
    files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith(".dcm")],
        key=lambda x: int(x.split("-")[-1].split(".")[0]),
    )

    count = len(files)
    if count == 0:
        return 0, 0, 0

    # Linear scan with stride to find boundaries efficiently
    stride = 3
    start_idx = 0
    end_idx = count - 1

    # Find start: Scan forward
    found_start = False
    for i in range(0, count, stride):
        path = os.path.join(directory_path, files[i])
        img = load_dicom_image(path)
        if img is not None and np.max(img) > 0:
            start_idx = i
            found_start = True
            break

    # Find end: Scan backward
    found_end = False
    for i in range(count - 1, -1, -stride):
        path = os.path.join(directory_path, files[i])
        img = load_dicom_image(path)
        if img is not None and np.max(img) > 0:
            end_idx = i
            found_end = True
            break

    # Fallback if scan missed (e.g., very small ROI) or empty images
    if not found_start:
        start_idx = 0
    if not found_end:
        end_idx = count - 1

    # Ensure consistency
    if end_idx < start_idx:
        start_idx, end_idx = 0, count - 1

    return start_idx, end_idx, count


def process_dataset_roi(df, load_cached_data=True):
    """
    Generates or loads ROI boundaries for the entire dataset.
    Implements caching mechanism using parquet.

    Args:
        df (pd.DataFrame): Metadata dataframe containing subject paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe containing ROI start/end/count for each modality.
    """
    cache_path = os.path.join(WORKING_DIR, "roi_boundaries_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Compute if cache missing or reload forced
    results = []
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        subject_res = {"BraTS21ID": sid}

        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            start, end, count = get_modality_roi(full_path)

            subject_res[f"{mod}_start"] = start
            subject_res[f"{mod}_end"] = end
            subject_res[f"{mod}_count"] = count

        results.append(subject_res)

    df_res = pd.DataFrame(results)

    # 3. Save to cache
    df_res.to_parquet(cache_path)

    return df_res

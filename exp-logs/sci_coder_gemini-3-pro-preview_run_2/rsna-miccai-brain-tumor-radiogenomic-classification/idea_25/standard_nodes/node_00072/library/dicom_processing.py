import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from library import config


def read_dicom_robust(path):
    """
    Reads a DICOM file with robust fallback strategies.
    1. Tries pydicom.dcmread
    2. Fallback: Raw binary tail-read assuming 512x512 uint16 (based on file size analysis).
    """
    # Strategy 1: Standard pydicom
    try:
        dcm = pydicom.dcmread(path)
        return dcm.pixel_array
    except Exception:
        pass

    # Strategy 2: Raw Binary Tail-Read
    # Based on file analysis, images are likely 512x512 (approx 525kB files)
    # 512 * 512 * 2 bytes = 524,288 bytes
    target_shape = (512, 512)
    expected_bytes = target_shape[0] * target_shape[1] * 2

    try:
        file_size = os.path.getsize(path)
        if file_size >= expected_bytes:
            with open(path, "rb") as f:
                # Seek to the end minus expected bytes
                f.seek(-expected_bytes, os.SEEK_END)
                raw_data = f.read(expected_bytes)

            img = np.frombuffer(raw_data, dtype=np.uint16)
            img = img.reshape(target_shape)
            return img
    except Exception:
        pass

    # Fallback return: Zero array to prevent pipeline crash
    return np.zeros(target_shape, dtype=np.uint16)


def preprocess_slice(image_data):
    """
    Preprocesses a single slice:
    1. Converts to float32
    2. Resizes to 224x224 using Area Interpolation
    3. Min-Max Normalization [0, 1]
    """
    img = image_data.astype(np.float32)

    # Resize if necessary
    if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
        img = cv2.resize(
            img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

    # Min-Max Normalization
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def select_roi_indices(flair_paths):
    """
    Calculates the Sum of Intensity on FLAIR images within the 15-85% depth range
    to find the anchor slice index.

    Args:
        flair_paths (list): List of file paths to FLAIR slices.

    Returns:
        int: The index of the anchor slice.
    """
    num_slices = len(flair_paths)
    if num_slices == 0:
        return 0

    # Define search range
    start_idx = int(num_slices * config.ROI_DEPTH_MIN)
    end_idx = int(num_slices * config.ROI_DEPTH_MAX)

    # Handle small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    max_intensity = -1.0
    anchor_idx = num_slices // 2  # Default to middle

    for i in range(start_idx, end_idx):
        path = flair_paths[i]
        try:
            img = read_dicom_robust(path)
            # Raw Sum of Intensity
            current_intensity = np.sum(img)

            if current_intensity > max_intensity:
                max_intensity = current_intensity
                anchor_idx = i
        except Exception:
            continue

    return anchor_idx


def get_stride_indices(anchor_index, total_slices):
    """
    Generates slice indices for a single stack with fixed stride.
    Handles boundary clamping (Cite solution_lesson_node_00062).

    Args:
        anchor_index (int): The center slice index.
        total_slices (int): Total number of slices in the volume.

    Returns:
        list: List of integers representing slice indices.
              Order: [Anchor-Stride, Anchor, Anchor+Stride]
    """
    indices = []

    # [Anchor-Stride, Anchor, Anchor+Stride]
    offsets = [-config.STRIDE, 0, config.STRIDE]
    for offset in offsets:
        idx = anchor_index + offset
        # Clamp to valid range
        idx = max(0, min(total_slices - 1, idx))
        indices.append(idx)

    return indices


def get_roi_anchor(subject_id, flair_paths, load_cached_data=True):
    """
    Retrieves the anchor index for a subject.
    Implements caching mechanism using parquet.

    Args:
        subject_id (int or str): The subject identifier.
        flair_paths (list): List of FLAIR slice paths.
        load_cached_data (bool): Whether to use the cache.

    Returns:
        int: The anchor slice index.
    """
    cache_file = os.path.join(config.WORKING_DIR, "roi_cache.parquet")
    subject_key = str(subject_id)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df_cache = pd.read_parquet(cache_file)
            if subject_key in df_cache["BraTS21ID"].values:
                # Return cached value
                row = df_cache[df_cache["BraTS21ID"] == subject_key]
                return int(row.iloc[0]["anchor_index"])
        except Exception:
            # If cache read fails, proceed to compute
            pass

    # 2. Compute from scratch
    anchor_idx = select_roi_indices(flair_paths)

    # 3. Save to cache (if caching is enabled)
    if load_cached_data:
        try:
            # Re-read to minimize overwrite window
            if os.path.exists(cache_file):
                df_cache = pd.read_parquet(cache_file)
            else:
                df_cache = pd.DataFrame(columns=["BraTS21ID", "anchor_index"])
                # Ensure types
                df_cache["BraTS21ID"] = df_cache["BraTS21ID"].astype(str)
                df_cache["anchor_index"] = df_cache["anchor_index"].astype(int)

            # Update or Append
            if subject_key in df_cache["BraTS21ID"].values:
                df_cache.loc[df_cache["BraTS21ID"] == subject_key, "anchor_index"] = (
                    anchor_idx
                )
            else:
                new_row = pd.DataFrame(
                    {"BraTS21ID": [subject_key], "anchor_index": [anchor_idx]}
                )
                df_cache = pd.concat([df_cache, new_row], ignore_index=True)

            df_cache.to_parquet(cache_file)
        except Exception:
            # Ignore write errors (e.g. race conditions)
            pass

    return anchor_idx

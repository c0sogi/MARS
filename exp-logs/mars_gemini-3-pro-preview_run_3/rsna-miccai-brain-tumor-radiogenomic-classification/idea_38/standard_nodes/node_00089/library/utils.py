import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import Config


def extract_slice_index(filename):
    """
    Extracts the integer slice index from a DICOM filename.
    Expected format: 'Image-10.dcm' -> 10.
    """
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_slice(rel_path, target_size=(224, 224)):
    """
    Reads a DICOM file from the relative path and returns the pixel array.
    Resizes the image to target_size.
    Returns a zero array if loading fails.
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)
    try:
        ds = pydicom.dcmread(full_path)
        img = ds.pixel_array.astype(np.float32)

        # Resize
        if img.shape != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        return img
    except Exception as e:
        # Return zero placeholder if read fails
        return np.zeros(target_size, dtype=np.float32)


def uniform_temporal_subsample(total_slices, num_samples):
    """
    Selects 'num_samples' indices uniformly from the 10%-90% range of 'total_slices'.
    """
    if total_slices == 0:
        return []

    # Define ROI range (10% to 90%)
    start = int(total_slices * 0.10)
    end = int(total_slices * 0.90)

    # Handle edge case where ROI is too small
    if end <= start:
        start = 0
        end = total_slices

    # Generate indices
    if total_slices < num_samples:
        # If fewer slices than samples, use all available and repeat/pad logic handled by caller
        # But here we just return indices, potentially with repeats if we use linspace on small range
        indices = np.linspace(start, max(end - 1, start), num_samples, dtype=int)
    else:
        indices = np.linspace(start, max(end - 1, start), num_samples, dtype=int)

    return indices


def normalize_modality_group(volume):
    """
    Applies Min-Max normalization to a 3D volume (D, H, W).
    Scales values to [0, 1].
    """
    if volume.size == 0:
        return volume

    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        return (volume - v_min) / (v_max - v_min)
    else:
        return np.zeros_like(volume)


def process_patient(row):
    """
    Process a single patient:
    1. Load paths for all 4 modalities.
    2. Subsample 32 slices per modality (10-90% depth).
    3. Normalize per modality.
    4. Split into Even/Odd streams.
    5. Stack into (2, 64, 224, 224).

    Returns:
        patient_data: np.ndarray of shape (2, 64, 224, 224)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Containers for the two streams
    # Stream 0: Even slices, Stream 1: Odd slices
    # Each stream will have 16 slices * 4 modalities = 64 channels
    stream_even_parts = []
    stream_odd_parts = []

    for mod in modalities:
        path_col = f"{mod}_paths"
        paths = row[path_col] if row[path_col] is not None else []

        # Sort paths numerically
        # We need to extract just the filename from the relative path for sorting
        paths = sorted(paths, key=lambda p: extract_slice_index(os.path.basename(p)))

        # Subsample indices
        indices = uniform_temporal_subsample(len(paths), Config.NUM_SLICES_TOTAL)

        # Load slices
        modality_volume = []
        for idx in indices:
            if idx < len(paths):
                img = load_dicom_slice(
                    paths[idx], target_size=(Config.IMG_SIZE, Config.IMG_SIZE)
                )
            else:
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            modality_volume.append(img)

        # Stack to (32, 224, 224)
        modality_volume = np.array(modality_volume, dtype=np.float32)

        # Normalize
        modality_volume = normalize_modality_group(modality_volume)

        # Split into Even (0, 2, ..) and Odd (1, 3, ..)
        # Slices 0, 2, ..., 30 -> indices 0::2
        even_slices = modality_volume[0::2]  # Shape (16, 224, 224)
        # Slices 1, 3, ..., 31 -> indices 1::2
        odd_slices = modality_volume[1::2]  # Shape (16, 224, 224)

        stream_even_parts.append(even_slices)
        stream_odd_parts.append(odd_slices)

    # Concatenate modalities along the channel dimension (dim 0 for the slice block)
    # Each part is (16, 224, 224). 4 parts -> (64, 224, 224)
    stream_even = np.concatenate(stream_even_parts, axis=0)
    stream_odd = np.concatenate(stream_odd_parts, axis=0)

    # Stack streams: (2, 64, 224, 224)
    patient_data = np.stack([stream_even, stream_odd], axis=0)

    return patient_data


def load_dataset(subset="train", load_cached_data=True):
    """
    Loads the dataset for the given subset ('train', 'val', 'test').
    Uses caching to speed up subsequent runs.

    Args:
        subset (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        X (np.ndarray): Shape (N, 2, 64, 224, 224)
        y (np.ndarray): Shape (N,) or None for test
        ids (np.ndarray): Shape (N,)
    """
    # Determine paths based on subset
    if subset == "train":
        meta_path = Config.TRAIN_META_PATH
        cache_X = Config.CACHE_TRAIN_X
        cache_ids = Config.CACHE_TRAIN_IDS
        cache_y = Config.CACHE_TRAIN_Y
    elif subset == "val":
        meta_path = Config.VAL_META_PATH
        cache_X = Config.CACHE_VAL_X
        cache_ids = Config.CACHE_VAL_IDS
        cache_y = Config.CACHE_VAL_Y
    elif subset == "test":
        meta_path = Config.TEST_META_PATH
        cache_X = Config.CACHE_TEST_X
        cache_ids = Config.CACHE_TEST_IDS
        cache_y = None  # No targets for test
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_X) and os.path.exists(cache_ids):
            if subset == "test" or os.path.exists(cache_y):
                print(f"Loading {subset} data from cache...")
                X = np.load(cache_X)
                ids = np.load(cache_ids)
                y = np.load(cache_y) if cache_y else None
                return X, y, ids

    # 2. Process from scratch
    print(f"Processing {subset} data from scratch...")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    # Debug mode: sample small subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process images
        patient_data = process_patient(row)
        X_list.append(patient_data)

        # Store ID
        ids_list.append(row["BraTS21ID"])

        # Store Target (if available)
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # 3. Save to cache
    print(f"Saving {subset} data to cache at {Config.CACHE_DIR}...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids

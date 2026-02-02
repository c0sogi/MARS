import os
import glob
import re
import numpy as np
import cv2
import pandas as pd
from library.config import (
    IMG_SIZE,
    RELATIVE_DEPTHS,
    MODALITIES,
    INPUT_DIR,
    seed_everything,
    INPUT_CHANNELS,
)


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm, Image-2.dcm, Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path):
    """
    Reads a DICOM file and returns a numpy array.
    Attempts to use pydicom first, then falls back to OpenCV.
    """
    # Attempt 1: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except (ImportError, Exception):
        pass

    # Attempt 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # Return None if failed
    return None


def normalize_slice(img):
    """
    Min-max scales the image to [0, 1].
    """
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = img - min_val  # Should be all zeros

    return img


def resize_slice(img, size=IMG_SIZE):
    """
    Resizes image to the target size.
    """
    if img is None:
        return np.zeros((size, size), dtype=np.float32)

    # Check if resize is needed
    if img.shape[0] == size and img.shape[1] == size:
        return img

    try:
        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img_resized
    except Exception:
        return np.zeros((size, size), dtype=np.float32)


def compute_roi_bounds(directory_path):
    """
    Scans a directory of DICOM files to find the start and end indices
    of slices containing actual brain tissue (max pixel value > 0).
    Returns (start_index, end_index, sorted_files).
    """
    if not os.path.exists(directory_path):
        return 0, 0, []

    files = glob.glob(os.path.join(directory_path, "*.dcm"))
    files.sort(key=natural_sort_key)

    if not files:
        return 0, 0, []

    # Linear scan to find bounds
    start_idx = -1
    end_idx = -1

    # We scan all files. To optimize, one might skip, but accuracy is priority here.
    for i, fpath in enumerate(files):
        img = read_dicom(fpath)
        if img is not None and np.max(img) > 0:
            if start_idx == -1:
                start_idx = i
            end_idx = i

    if start_idx == -1:
        # No signal found, default to middle
        mid = len(files) // 2
        return mid, mid, files

    return start_idx, end_idx, files


def get_relative_indices(start_idx, end_idx, depths=RELATIVE_DEPTHS):
    """
    Calculates the file indices corresponding to the relative depths within the ROI.
    """
    roi_len = end_idx - start_idx + 1
    indices = []
    for d in depths:
        # Calculate offset
        offset = int(d * roi_len)
        # Determine index
        idx = start_idx + offset
        # Clip to bounds
        idx = max(start_idx, min(end_idx, idx))
        indices.append(idx)
    return indices


def process_subject(row):
    """
    Processes a single subject to create the 9-channel input tensor.
    Structure:
    - Depth 40%: [FLAIR, T1wCE, T2w]
    - Depth 50%: [FLAIR, T1wCE, T2w]
    - Depth 60%: [FLAIR, T1wCE, T2w]

    Returns: numpy array of shape (IMG_SIZE, IMG_SIZE, 9)
    """
    # Initialize storage for the 3 depths x 3 modalities
    # We want final order:
    # Ch 0-2: Depth 0 (Mod 0, Mod 1, Mod 2)
    # Ch 3-5: Depth 1 (Mod 0, Mod 1, Mod 2)
    # Ch 6-8: Depth 2 (Mod 0, Mod 1, Mod 2)

    # Temporary storage: slices[depth_idx][modality_idx]
    slices_grid = [
        [None for _ in range(len(MODALITIES))] for _ in range(len(RELATIVE_DEPTHS))
    ]

    for m_idx, mod in enumerate(MODALITIES):
        # Construct path (metadata contains relative path)
        rel_path = row[f"{mod.lower()}_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # 1. Compute ROI
        start, end, files = compute_roi_bounds(full_path)

        if not files:
            # Handle missing directory by filling with zeros later
            continue

        # 2. Get Indices
        indices = get_relative_indices(start, end, RELATIVE_DEPTHS)

        # 3. Read and Process Slices
        for d_idx, file_idx in enumerate(indices):
            fpath = files[file_idx]
            img = read_dicom(fpath)
            img = normalize_slice(img)
            img = resize_slice(img, IMG_SIZE)
            slices_grid[d_idx][m_idx] = img

    # Flatten into 9 channels
    final_channels = []
    for d_idx in range(len(RELATIVE_DEPTHS)):
        for m_idx in range(len(MODALITIES)):
            img = slices_grid[d_idx][m_idx]
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            final_channels.append(img)

    # Stack along last axis -> (H, W, C)
    # Shape: (224, 224, 9)
    vol = np.stack(final_channels, axis=-1)
    return vol


def process_dataset(
    metadata_df,
    cache_ids_path,
    cache_images_path,
    cache_labels_path=None,
    load_cached_data=True,
    debug=False,
):
    """
    Processes the entire dataset defined in metadata_df.
    Handles caching to .npy files.

    Args:
        metadata_df: DataFrame containing subject paths.
        cache_ids_path: Path to save/load IDs.
        cache_images_path: Path to save/load Images.
        cache_labels_path: Path to save/load Labels (optional).
        load_cached_data: Boolean, whether to attempt loading from cache.
        debug: Boolean, if True, process only a small subset.

    Returns:
        ids (np.array), images (np.array), labels (np.array or None)
    """

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_images_path), exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data:
        try:
            if os.path.exists(cache_images_path) and os.path.exists(cache_ids_path):
                # Check labels if path provided
                if cache_labels_path and not os.path.exists(cache_labels_path):
                    raise FileNotFoundError("Labels cache missing")

                print(
                    f"Loading cached data from {os.path.dirname(cache_images_path)}..."
                )
                ids = np.load(cache_ids_path)
                images = np.load(cache_images_path)
                labels = np.load(cache_labels_path) if cache_labels_path else None
                return ids, images, labels
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing data...")

    # 2. Process from Scratch
    print("Processing dataset from scratch...")

    processed_ids = []
    processed_images = []
    processed_labels = []

    # Handle debug mode
    df_to_process = metadata_df.head(50) if debug else metadata_df

    total = len(df_to_process)

    for idx, row in df_to_process.iterrows():
        sid = row["BraTS21ID"]

        # Process image volume
        vol = process_subject(row)  # (224, 224, 9)

        processed_ids.append(sid)
        processed_images.append(vol)

        if "MGMT_value" in row:
            processed_labels.append(row["MGMT_value"])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} subjects")

    # Convert to numpy arrays
    ids_arr = np.array(processed_ids)
    # Stack images: (N, 224, 224, 9)
    images_arr = np.array(processed_images, dtype=np.float32)

    if processed_labels:
        labels_arr = np.array(processed_labels, dtype=np.float32)
    else:
        labels_arr = None

    # 3. Save to Cache
    print(f"Saving processed data to {os.path.dirname(cache_images_path)}...")
    np.save(cache_ids_path, ids_arr)
    np.save(cache_images_path, images_arr)
    if labels_arr is not None and cache_labels_path:
        np.save(cache_labels_path, labels_arr)

    return ids_arr, images_arr, labels_arr

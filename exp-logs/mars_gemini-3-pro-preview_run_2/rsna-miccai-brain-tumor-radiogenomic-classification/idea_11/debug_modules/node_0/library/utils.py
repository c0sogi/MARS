import os
import re
import numpy as np
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from library.config import Config


def get_image_plane(file_name):
    """
    Extracts the slice number from the filename (e.g., 'Image-10.dcm' -> 10).
    Used for sorting DICOM files anatomically.
    """
    match = re.search(r"Image-(\d+)", file_name)
    if match:
        return int(match.group(1))
    return -1


def read_dicom_robust(file_path):
    """
    Reads a DICOM file robustly to handle corrupt headers.

    Strategy:
    1. Attempt standard read using OpenCV.
    2. If that fails, fallback to raw binary tail-read based on file size.
    """
    # 1. Try OpenCV (Standard Read)
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    # 2. Binary Fallback (Tail-Read)
    # This handles cases where the DICOM header is corrupt but pixel data is intact.
    try:
        file_size = os.path.getsize(file_path)

        # Infer dimensions from file size
        # 512x512 uint16 = 524,288 bytes
        # 256x256 uint16 = 131,072 bytes
        # We check if file is large enough to contain the raw data

        if file_size >= 524288:
            shape = (512, 512)
            num_pixels = 512 * 512
        elif file_size >= 131072:
            shape = (256, 256)
            num_pixels = 256 * 256
        else:
            # File too small to contain a standard MRI slice
            return None

        expected_bytes = num_pixels * 2  # uint16 = 2 bytes per pixel

        with open(file_path, "rb") as f:
            # Seek to the end minus the pixel data size
            f.seek(-expected_bytes, os.SEEK_END)
            data = f.read(expected_bytes)

        img_array = np.frombuffer(data, dtype=np.uint16)
        img_array = img_array.reshape(shape)
        return img_array

    except Exception:
        # If both methods fail, return None to be handled by the caller
        return None


def preprocess_image(img, target_size=(256, 256)):
    """
    Preprocesses the raw image array for the model.

    Steps:
    1. Convert to float32 (Precision Preservation).
    2. Resize using Area Interpolation (Noise Suppression).
    3. Min-Max Normalize to [0, 1].
    """
    if img is None:
        return np.zeros(target_size, dtype=np.float32)

    # Convert to float32 to avoid quantization errors during processing
    img = img.astype(np.float32)

    # Resize if necessary
    # cv2.INTER_AREA is preferred for downsampling as it suppresses moire/noise
    if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Min-Max Normalization
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def _process_subject_roi(row):
    """
    Worker function to calculate the ROI anchor slice for a single subject.
    Scans FLAIR images to find the slice with maximum signal intensity.
    """
    subject_id = row["BraTS21ID"]
    flair_path_rel = row["path_FLAIR"]
    flair_dir = os.path.join(Config.INPUT_DIR, flair_path_rel)

    if not os.path.exists(flair_dir):
        return subject_id, 0

    files = os.listdir(flair_dir)
    # Filter for valid image files and sort anatomically
    dicom_files = [f for f in files if "Image-" in f]
    dicom_files.sort(key=get_image_plane)

    if not dicom_files:
        return subject_id, 0

    num_slices = len(dicom_files)

    # Define the search range (e.g., 15% to 85% of depth)
    start_idx = int(num_slices * Config.ROI_SEARCH_MIN)
    end_idx = int(num_slices * Config.ROI_SEARCH_MAX)

    # Handle small volumes where range might be invalid
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    intensities = []
    valid_indices = []

    # Read slices in the search range and calculate mean intensity
    for i in range(start_idx, end_idx):
        f_path = os.path.join(flair_dir, dicom_files[i])
        img = read_dicom_robust(f_path)

        if img is not None:
            intensities.append(np.mean(img))
        else:
            intensities.append(0.0)
        valid_indices.append(i)

    if not intensities:
        return subject_id, num_slices // 2

    # Apply smoothing to the intensity profile to avoid artifacts/spikes
    intensities = np.array(intensities)
    window = Config.SMOOTHING_WINDOW
    if len(intensities) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(intensities, kernel, mode="same")
    else:
        smoothed = intensities

    # Find the index of the peak intensity
    peak_local_idx = np.argmax(smoothed)
    best_slice_idx = valid_indices[peak_local_idx]

    return subject_id, best_slice_idx


def get_roi_indices(metadata_df, load_cached_data=True):
    """
    Determines the anchor slice index for each subject in the metadata.

    This process is computationally expensive (reading thousands of DICOMs),
    so it implements a caching mechanism using Parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")

    # 1. Attempt to Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            roi_map = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))

            # Verify cache covers all requested IDs
            missing_ids = [
                pid for pid in metadata_df["BraTS21ID"] if pid not in roi_map
            ]
            if not missing_ids:
                print(f"Loaded ROI indices from cache: {cache_path}")
                return roi_map
            else:
                print("Cache incomplete. Recomputing ROI indices...")
        except Exception as e:
            print(f"Failed to load ROI cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print("Computing ROI indices (scanning FLAIR volumes)...")

    # Prepare rows for parallel processing
    rows = metadata_df.to_dict("records")
    results = {}

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        futures = executor.map(_process_subject_roi, rows)

        for pid, idx in futures:
            results[pid] = idx

    # 3. Save Cache
    try:
        cache_df = pd.DataFrame(
            list(results.items()), columns=["BraTS21ID", "anchor_index"]
        )
        cache_df.to_parquet(cache_path)
        print(f"Saved ROI indices to cache: {cache_path}")
    except Exception as e:
        print(f"Failed to save ROI cache: {e}")

    return results

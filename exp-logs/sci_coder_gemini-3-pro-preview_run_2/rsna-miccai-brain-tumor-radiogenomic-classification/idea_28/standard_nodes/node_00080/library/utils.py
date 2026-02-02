import os
import numpy as np
import cv2
import pandas as pd
import re
from library.config import Config


def read_dicom_robust(path):
    """
    Reads a DICOM file using a Raw Binary Tail-Read strategy to bypass brittle headers.
    Assumes Little Endian uint16 data.
    Handles standard MRI resolutions (512x512, 256x256).
    """
    try:
        with open(path, "rb") as f:
            content = f.read()
    except Exception:
        # Return a blank image if file reading fails completely
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    file_size = len(content)

    # MRI images in this dataset are typically uint16 (2 bytes per pixel)
    # Check for 512x512
    size_512 = 512 * 512 * 2
    if file_size >= size_512:
        # Read the last N bytes
        pixel_data = content[-size_512:]
        try:
            img = np.frombuffer(pixel_data, dtype=np.uint16).reshape(512, 512)
            return img.astype(np.float32)
        except Exception:
            pass

    # Check for 256x256
    size_256 = 256 * 256 * 2
    if file_size >= size_256:
        pixel_data = content[-size_256:]
        try:
            img = np.frombuffer(pixel_data, dtype=np.uint16).reshape(256, 256)
            return img.astype(np.float32)
        except Exception:
            pass

    # Fallback: If dimensions don't match standard buckets, return zeros
    # This prevents the pipeline from crashing on a single corrupt file
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def resize_image(image, size=Config.IMG_SIZE):
    """
    Resizes an image to the target size using Area Interpolation.
    Area interpolation is preferred for downsampling to reduce aliasing.
    """
    if image is None or image.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    try:
        # cv2.INTER_AREA is best for shrinking images (moire-free)
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return np.zeros((size, size), dtype=np.float32)


def normalize_min_max(image):
    """
    Applies Independent Per-Channel Min-Max Scaling to [0, 1].
    """
    if image is None:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    img_min = np.min(image)
    img_max = np.max(image)

    if img_max - img_min > 1e-6:
        return (image - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(image, dtype=np.float32)


def get_sorted_image_files(folder_path):
    """
    Returns a sorted list of DICOM files in a directory.
    Sorts numerically based on the 'Image-N.dcm' pattern.
    """
    if not os.path.exists(folder_path):
        return []

    files = os.listdir(folder_path)
    # Filter for .dcm files
    files = [f for f in files if f.endswith(".dcm")]

    # Sort by the integer number in the filename
    def extract_number(filename):
        match = re.search(r"Image-(\d+)", filename)
        return int(match.group(1)) if match else 0

    return sorted(files, key=extract_number)


def select_roi_indices(subject_id, paths_dict):
    """
    Implements Integral-Statistic ROI Selection.
    Cite solution_lesson_node_00038: Prefer integral statistics (Sum) over Extremal (Max).
    Cite solution_lesson_node_00053: Derive selection from a single dominant reference modality.
    Cite solution_lesson_node_00026: Minimize hard constraints (no arbitrary safe zones).

    Returns:
        int: The selected anchor slice index (relative to the sorted file list).
    """
    # Order of preference for determining geometry and ROI
    # FLAIR is the dominant modality for edema/tumor bulk
    modalities = ["path_FLAIR", "path_T1w", "path_T1wCE", "path_T2w"]

    selected_modality = None
    files = []
    dir_path = ""

    # Find the first available modality to act as reference
    for mod in modalities:
        if mod in paths_dict:
            p = os.path.join(Config.INPUT_DIR, paths_dict[mod])
            f = get_sorted_image_files(p)
            if len(f) > 0:
                selected_modality = mod
                files = f
                dir_path = p
                break

    if not files:
        return 0

    # Calculate sum of pixel intensities for every slice
    # This identifies the slice with the most "brain tissue" (or tumor/edema signal in FLAIR)
    # robustly handling noise that might trigger a 'Max' heuristic.
    sums = []
    for f in files:
        img = read_dicom_robust(os.path.join(dir_path, f))
        sums.append(np.sum(img))

    # Select the slice with the maximum total intensity
    best_idx = np.argmax(sums)

    return int(best_idx)


def generate_roi_cache(metadata_df, load_cached_data=True):
    """
    Generates or loads the ROI anchor indices for the dataset.
    Uses Parquet for caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing subject IDs and paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of {str(BraTS21ID): int(anchor_index)}
    """
    cache_path = Config.CACHE_FILE_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dictionary
            cache_dict = pd.Series(
                cache_df.anchor_index.values, index=cache_df.BraTS21ID.astype(str)
            ).to_dict()
            return cache_dict
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    results = []

    # Iterate over metadata
    # (No progress bar as per instructions)
    for idx, row in metadata_df.iterrows():
        subject_id = str(row["BraTS21ID"])

        # Extract paths
        paths = {
            "path_FLAIR": row["path_FLAIR"],
            "path_T1w": row["path_T1w"],
            "path_T1wCE": row["path_T1wCE"],
            "path_T2w": row["path_T2w"],
        }

        anchor_idx = select_roi_indices(subject_id, paths)

        results.append({"BraTS21ID": subject_id, "anchor_index": anchor_idx})

    # 3. Save to cache
    result_df = pd.DataFrame(results)
    # Ensure ID is string for consistency
    result_df["BraTS21ID"] = result_df["BraTS21ID"].astype(str)
    result_df.to_parquet(cache_path, index=False)

    # Convert to dict for return
    cache_dict = pd.Series(
        result_df.anchor_index.values, index=result_df.BraTS21ID
    ).to_dict()

    return cache_dict

import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_raw(path):
    """
    Reads a DICOM file using the Raw Binary Tail-Read strategy.

    This function assumes the pixel data is uncompressed and located at the
    end of the file. It infers the resolution (512x512 or 256x256) based on
    the file size to determine how many bytes to read.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array of type uint16. Returns a blank array on failure.
    """
    try:
        file_size = os.path.getsize(path)

        # Define expected byte sizes for standard uint16 resolutions
        # 512 * 512 * 2 bytes = 524,288 bytes
        # 256 * 256 * 2 bytes = 131,072 bytes
        size_512 = 512 * 512 * 2
        size_256 = 256 * 256 * 2

        # Determine resolution based on file size (file size = header + pixel data)
        if file_size >= size_512:
            rows, cols = 512, 512
            num_bytes = size_512
        elif file_size >= size_256:
            rows, cols = 256, 256
            num_bytes = size_256
        else:
            # File is too small to contain a standard image
            return np.zeros((256, 256), dtype=np.uint16)

        with open(path, "rb") as f:
            # Seek to the start of the pixel data (end of file - image bytes)
            f.seek(-num_bytes, 2)
            data = f.read(num_bytes)

        # Convert binary data to numpy array
        img = np.frombuffer(data, dtype=np.uint16).reshape((rows, cols))
        return img

    except Exception:
        # Return a blank image in case of I/O errors to prevent pipeline crash
        return np.zeros((256, 256), dtype=np.uint16)


def get_flair_roi_index(subject_dir):
    """
    Calculates the anchor slice index for a subject based on the FLAIR modality.

    Iterates through slices within the configured depth bounds and selects the
    slice with the highest sum of raw pixel intensity.

    Args:
        subject_dir (str): Path to the FLAIR directory for a subject.

    Returns:
        int: The index of the anchor slice relative to the sorted file list.
    """
    if not os.path.exists(subject_dir):
        return 0

    # List files and sort numerically (Image-1.dcm, Image-2.dcm, ...)
    files = [f for f in os.listdir(subject_dir) if f.endswith(".dcm")]
    if not files:
        return 0

    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except Exception:
        files.sort()  # Fallback to lexicographical sort if naming differs

    num_slices = len(files)
    if num_slices == 0:
        return 0

    # Determine search bounds (e.g., 15% to 85% of volume depth)
    start_idx = int(num_slices * Config.ROI_BOUNDS[0])
    end_idx = int(num_slices * Config.ROI_BOUNDS[1])

    # Clamp bounds to valid range
    start_idx = max(0, start_idx)
    end_idx = min(num_slices, end_idx)
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    max_intensity = -1.0
    best_index = start_idx

    # Iterate through the defined range to find the slice with max signal
    for i in range(start_idx, end_idx):
        f_path = os.path.join(subject_dir, files[i])
        img = read_dicom_raw(f_path)

        # Calculate Sum of Intensity on raw pixels
        current_intensity = np.sum(img)

        if current_intensity > max_intensity:
            max_intensity = current_intensity
            best_index = i

    return best_index


def get_roi_map(metadata_df, load_cached_data=True):
    """
    Generates or loads a mapping of BraTS21ID to the best FLAIR slice index.

    Implements a caching mechanism using Parquet. It loads existing cache data,
    computes indices for any new subjects found in metadata_df, and updates
    the cache file.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'BraTS21ID' and 'path_FLAIR'.
        load_cached_data (bool): If True, attempts to load from existing cache.
                                 If False, forces recomputation for subjects in metadata_df.

    Returns:
        dict: Mapping {BraTS21ID (int): anchor_index (int)}
    """
    cache_path = Config.CACHE_ROI_PATH
    roi_map = {}

    # 1. Load existing cache if available
    if os.path.exists(cache_path):
        try:
            df_existing = pd.read_parquet(cache_path)
            if (
                "BraTS21ID" in df_existing.columns
                and "anchor_index" in df_existing.columns
            ):
                roi_map = dict(
                    zip(df_existing["BraTS21ID"], df_existing["anchor_index"])
                )
        except Exception:
            # If cache is corrupt, start fresh
            pass

    # 2. Identify subjects that need computation
    needed_ids = metadata_df["BraTS21ID"].unique()

    # If load_cached_data is False, we treat all needed_ids as missing to force update
    missing_ids = [
        uid for uid in needed_ids if uid not in roi_map or not load_cached_data
    ]

    if not missing_ids:
        return roi_map

    # 3. Compute indices for missing subjects
    # Create a lookup for paths to avoid repeated DataFrame filtering
    id_to_path = metadata_df.set_index("BraTS21ID")["path_FLAIR"].to_dict()

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    for subject_id in missing_ids:
        rel_path = id_to_path.get(subject_id)
        if rel_path:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            anchor_idx = get_flair_roi_index(full_path)
        else:
            anchor_idx = 0

        # Update the map
        roi_map[subject_id] = anchor_idx

    # 4. Save updated cache to disk
    try:
        # Convert dict back to DataFrame for saving
        save_data = [{"BraTS21ID": k, "anchor_index": v} for k, v in roi_map.items()]
        pd.DataFrame(save_data).to_parquet(cache_path)
    except Exception:
        # Non-critical failure (e.g., disk full), proceed with in-memory map
        pass

    return roi_map

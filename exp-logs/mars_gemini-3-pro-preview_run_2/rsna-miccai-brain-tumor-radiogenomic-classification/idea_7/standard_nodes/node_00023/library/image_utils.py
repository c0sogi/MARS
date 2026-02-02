import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
from library.utils import load_metadata

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_7"
CACHE_FILE = "roi_cache.parquet"
IMG_SIZE = (256, 256)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_robust(path):
    """
    Reads a DICOM file. Tries OpenCV first, falls back to raw binary reading
    based on file size heuristics (assuming 16-bit depth).
    """
    # Attempt 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # Attempt 2: Raw Binary Fallback
    # Heuristic: BraTS data is typically 256x256 or 512x512, uint16 (2 bytes per pixel)
    try:
        file_size = os.path.getsize(path)

        # Expected data sizes
        size_256 = 256 * 256 * 2  # 131,072 bytes
        size_512 = 512 * 512 * 2  # 524,288 bytes

        pixel_data = None
        shape = None

        if file_size >= size_512:
            offset = file_size - size_512
            shape = (512, 512)
            with open(path, "rb") as f:
                f.seek(offset)
                pixel_data = f.read()
        elif file_size >= size_256:
            offset = file_size - size_256
            shape = (256, 256)
            with open(path, "rb") as f:
                f.seek(offset)
                pixel_data = f.read()

        if pixel_data is not None and shape is not None:
            arr = np.frombuffer(pixel_data, dtype=np.uint16)
            if arr.size == shape[0] * shape[1]:
                return arr.reshape(shape)
    except Exception:
        pass

    # Fallback: Return a black image of default size to prevent pipeline crash
    return np.zeros(IMG_SIZE, dtype=np.uint16)


def compute_flair_anchor(flair_dir_path):
    """
    Computes the anchor slice index for a patient based on the FLAIR modality.
    Logic:
    1. Read all slices, compute mean intensity.
    2. Apply Moving Average (window=5).
    3. Exclude top/bottom 15%.
    4. Find index of max value in smoothed profile.
    """
    if not os.path.exists(flair_dir_path):
        return 0

    files = sorted(os.listdir(flair_dir_path), key=natural_sort_key)
    if not files:
        return 0

    intensities = []

    # Read every slice to build profile
    for f in files:
        f_path = os.path.join(flair_dir_path, f)
        img = read_dicom_robust(f_path)
        # Cite solution_lesson_node_00019: Summing intensity is more robust than mean for ROI selection
        intensities.append(np.sum(img.astype(np.float32)))

    if not intensities:
        return 0

    # Z-Axis Smoothing
    series = pd.Series(intensities)
    smoothed = (
        series.rolling(window=5, center=True, min_periods=1).mean().fillna(0).values
    )

    # Boundary Constraints
    n_slices = len(smoothed)
    start_idx = int(n_slices * 0.15)
    end_idx = int(n_slices * 0.85)

    # Safety check for very small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_slices

    # Find peak in the valid range
    valid_range = smoothed[start_idx:end_idx]
    if len(valid_range) == 0:
        return n_slices // 2

    local_argmax = np.argmax(valid_range)
    anchor_index = start_idx + local_argmax

    return int(anchor_index)


def get_roi_cache(metadata_df, load_cached_data=True):
    """
    Manages the cache of computed anchor indices.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)

    cache_df = pd.DataFrame(columns=["BraTS21ID", "anchor_index"])

    # Load existing cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded_df = pd.read_parquet(cache_path)
            # Validate schema to guard against invalid cache states (Cite debug_lesson_1)
            if "BraTS21ID" in loaded_df.columns and "anchor_index" in loaded_df.columns:
                cache_df = loaded_df
        except Exception:
            pass  # Corrupt cache, start fresh

    # Determine which IDs need processing
    # metadata_df might have BraTS21ID as int, ensure consistency
    metadata_ids = metadata_df["BraTS21ID"].unique()
    cached_ids = cache_df["BraTS21ID"].unique()

    missing_ids = [mid for mid in metadata_ids if mid not in cached_ids]

    if missing_ids:
        new_entries = []
        # Process missing IDs
        # We need to look up the path for each missing ID from metadata_df
        # Create a lookup dict for speed
        path_lookup = metadata_df.set_index("BraTS21ID")["path_FLAIR"].to_dict()

        for mid in missing_ids:
            rel_path = path_lookup.get(mid)
            if rel_path:
                full_path = os.path.join(INPUT_DIR, rel_path)
                anchor = compute_flair_anchor(full_path)
                new_entries.append({"BraTS21ID": mid, "anchor_index": anchor})
            else:
                new_entries.append({"BraTS21ID": mid, "anchor_index": 0})

        # Update cache
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            # Save updated cache
            cache_df.to_parquet(cache_path, index=False)

    # Convert to dictionary for O(1) access
    return cache_df.set_index("BraTS21ID")["anchor_index"].to_dict()


def load_patient_volume(row, anchor_index, target_size=IMG_SIZE):
    """
    Loads the 12-channel volume for a patient.

    Args:
        row: Series/dict containing paths (path_FLAIR, path_T1w, etc.)
        anchor_index: The index of the anchor slice in the FLAIR modality.
        target_size: Tuple (H, W) for resizing.

    Returns:
        torch.Tensor: Shape (12, H, W), float32, normalized 0-1.
    """
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    channels = []

    # 1. Determine FLAIR context to map to other modalities
    flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
    flair_files = (
        sorted(os.listdir(flair_path), key=natural_sort_key)
        if os.path.exists(flair_path)
        else []
    )
    n_flair = len(flair_files)

    # Calculate relative position of the anchor (0.0 to 1.0)
    # If n_flair is 0, default to 0.5
    rel_pos = anchor_index / n_flair if n_flair > 0 else 0.5

    for mod in modalities:
        mod_dir = os.path.join(INPUT_DIR, row[f"path_{mod}"])
        files = (
            sorted(os.listdir(mod_dir), key=natural_sort_key)
            if os.path.exists(mod_dir)
            else []
        )
        n_files = len(files)

        if n_files == 0:
            # Missing modality: return black channels
            for _ in range(3):
                channels.append(np.zeros(target_size, dtype=np.float32))
            continue

        # Map relative position to this modality's slice index
        # This handles variable slice counts (co-registration assumption)
        center_idx = int(rel_pos * n_files)

        # Select 3 slices: Center-5, Center, Center+5
        # Clamp to valid range [0, n_files-1]
        indices = [center_idx - 5, center_idx, center_idx + 5]
        indices = [max(0, min(idx, n_files - 1)) for idx in indices]

        for idx in indices:
            file_path = os.path.join(mod_dir, files[idx])
            img = read_dicom_robust(file_path)

            # Resize
            if img.shape != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

            # Normalize per slice/channel to 0-1
            # Robust min-max
            img = img.astype(np.float32)
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros_like(img)

            channels.append(img)

    # Stack channels
    # Result shape: (12, H, W)
    volume = np.stack(channels, axis=0)
    return torch.tensor(volume, dtype=torch.float32)

import os
import re
import numpy as np
import pandas as pd
import cv2
import torch

# Import configuration and utilities
from library.config import (
    IMG_SIZE,
    DEPTH_OFFSET,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
)
from library.utils import load_or_process_cache

# Attempt to import pydicom for robust DICOM reading, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Extracts the integer instance number from the filename (e.g., 'Image-10.dcm' -> 10).
    Used to sort DICOM files spatially.
    """
    match = re.search(r"Image-(\d+)", s)
    return int(match.group(1)) if match else 0


def read_dicom_sequence(directory):
    """
    Returns a sorted list of full file paths for DICOM files in a directory.
    Files are sorted by the instance number in their filename to ensure spatial continuity.
    """
    if not os.path.exists(directory):
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".dcm")]
    files.sort(key=natural_sort_key)
    return [os.path.join(directory, f) for f in files]


def read_image_file(path):
    """
    Reads a DICOM file and returns the pixel array.
    Prioritizes pydicom if available, otherwise falls back to OpenCV.
    """
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            return dcm.pixel_array
        except Exception:
            pass

    # Fallback to OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def compute_brain_roi(file_paths):
    """
    Computes the Z-axis Center of Mass (CoM) and Depth of the brain tissue.
    Iterates through the volume to identify slices containing brain tissue (pixels > 0).

    Returns:
        com_z (float): The mean index of slices containing brain signal.
        depth (int): The number of slices between the first and last brain signal.
        count (int): Total number of files in the sequence.
    """
    if not file_paths:
        return 0, 0, 0

    z_indices = []

    # Iterate through all files to find brain presence
    for i, path in enumerate(file_paths):
        img = read_image_file(path)
        # Check for signal (max > 0)
        if img is not None and np.max(img) > 0:
            z_indices.append(i)

    if not z_indices:
        # Fallback if no signal found: return middle index
        count = len(file_paths)
        return count // 2, 0, count

    z_indices = np.array(z_indices)
    min_z = z_indices[0]
    max_z = z_indices[-1]
    depth = max_z - min_z

    # Center of Mass of the indices (geometric center of the occupied slices)
    com_z = np.mean(z_indices)

    return com_z, depth, len(file_paths)


def _generate_stats_cache(metadata_df):
    """
    Internal function to compute stats for all subjects in the dataframe.
    This is the heavy-lifting function passed to load_or_process_cache.
    """
    stats = []
    modalities = ["flair", "t1wce", "t2w"]

    print(f"Computing volumetric stats for {len(metadata_df)} subjects...")

    for idx, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]
        subject_stats = {"BraTS21ID": sid}

        for mod in modalities:
            # Metadata paths are relative to input dir
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            paths = read_dicom_sequence(full_path)
            com, depth, count = compute_brain_roi(paths)

            subject_stats[f"{mod}_com"] = com
            subject_stats[f"{mod}_depth"] = depth
            subject_stats[f"{mod}_count"] = count

        stats.append(subject_stats)

    return pd.DataFrame(stats)


def get_subject_stats(metadata_df, split_name, load_cached_data=True):
    """
    Retrieves or computes the CoM and Depth statistics for the given dataset split.
    Uses caching to avoid re-reading DICOMs on subsequent runs.
    """
    cache_file = os.path.join(CACHE_DIR, f"stats_{split_name}.parquet")

    return load_or_process_cache(
        cache_path=cache_file,
        process_fn=_generate_stats_cache,
        load_cached_data=load_cached_data,
        metadata_df=metadata_df,
    )


def select_slice_indices(com, depth, count, offset_factor=DEPTH_OFFSET):
    """
    Calculates indices for -Offset, Center, and +Offset relative to the CoM.
    Ensures indices are within bounds [0, count-1].
    """
    if count == 0:
        return 0, 0, 0

    indices = []
    # Offsets: -10%, 0%, +10% (controlled by offset_factor)
    multipliers = [-1.0, 0.0, 1.0]

    for m in multipliers:
        idx = com + (m * offset_factor * depth)
        # Round to nearest integer and clamp
        idx = int(np.round(idx))
        idx = max(0, min(idx, count - 1))
        indices.append(idx)

    return indices


def process_image(img):
    """
    Normalizes image to [0, 1] and resizes to target IMG_SIZE.
    """
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    img = img.astype(np.float32)

    # Min-Max Normalization
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    # Resize
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    return img


def load_subject_volume(subject_row, stats_row):
    """
    Loads the 9-channel volume for a subject based on precomputed stats.

    Channel Architecture:
      0-2: [FLAIR, T1wCE, T2w] at -10% depth
      3-5: [FLAIR, T1wCE, T2w] at CoM
      6-8: [FLAIR, T1wCE, T2w] at +10% depth

    Returns:
        torch.Tensor: Shape (9, IMG_SIZE, IMG_SIZE)
    """
    modalities = ["flair", "t1wce", "t2w"]

    # Container for all slices: slices_by_modality[mod] = [img_low, img_mid, img_high]
    slices_by_modality = {}

    for mod in modalities:
        # Get paths
        rel_path = subject_row[f"{mod}_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)
        paths = read_dicom_sequence(full_path)

        # Get stats
        com = stats_row[f"{mod}_com"]
        depth = stats_row[f"{mod}_depth"]
        count = stats_row[f"{mod}_count"]

        # Determine indices
        indices = select_slice_indices(com, depth, count)

        mod_imgs = []
        for idx in indices:
            if idx < len(paths):
                img_raw = read_image_file(paths[idx])
                img_proc = process_image(img_raw)
            else:
                img_proc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            mod_imgs.append(img_proc)

        slices_by_modality[mod] = mod_imgs

    # Stack into final tensor (Channels, H, W)
    final_channels = []

    # Low (-10%)
    final_channels.append(slices_by_modality["flair"][0])
    final_channels.append(slices_by_modality["t1wce"][0])
    final_channels.append(slices_by_modality["t2w"][0])

    # Mid (CoM)
    final_channels.append(slices_by_modality["flair"][1])
    final_channels.append(slices_by_modality["t1wce"][1])
    final_channels.append(slices_by_modality["t2w"][1])

    # High (+10%)
    final_channels.append(slices_by_modality["flair"][2])
    final_channels.append(slices_by_modality["t1wce"][2])
    final_channels.append(slices_by_modality["t2w"][2])

    # Stack
    tensor_np = np.stack(final_channels, axis=0)

    return torch.tensor(tensor_np, dtype=torch.float32)


def get_processed_dataframe(split="train", load_cached_data=True):
    """
    Helper function to load metadata and merge it with computed stats.
    This is the main entry point for the Dataset class.
    """
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError("Unknown split")

    df = pd.read_csv(meta_path)

    # Compute/Load stats
    stats_df = get_subject_stats(df, split, load_cached_data=load_cached_data)

    # Merge on BraTS21ID
    df = df.merge(stats_df, on="BraTS21ID", how="left")

    return df

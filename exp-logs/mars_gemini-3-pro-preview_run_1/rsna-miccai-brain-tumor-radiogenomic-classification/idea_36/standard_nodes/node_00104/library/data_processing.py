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


def _generate_stats_cache(metadata_df):
    """
    Internal function to compute stats for all subjects in the dataframe.
    Simplified to just count files as we use naive middle slice selection.
    Cite solution_lesson_node_00036
    """
    stats = []
    modalities = ["flair", "t1wce", "t2w"]

    print(f"Computing basic stats for {len(metadata_df)} subjects...")

    for idx, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]
        subject_stats = {"BraTS21ID": sid}

        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Just counting files is fast
            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                count = len(files)
            else:
                count = 0

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
    Loads the 3-channel volume (Middle Slice) for a subject.
    Cite solution_lesson_node_00002: Naive middle slice selection.
    Cite solution_lesson_node_00036: Independent heuristics per modality.
    Cite solution_lesson_node_00031: Early Fusion (Channel Stacking).

    Channel Architecture:
      0: FLAIR (Middle)
      1: T1wCE (Middle)
      2: T2w (Middle)

    Returns:
        torch.Tensor: Shape (3, IMG_SIZE, IMG_SIZE)
    """
    modalities = ["flair", "t1wce", "t2w"]
    final_channels = []

    for mod in modalities:
        # Get paths
        rel_path = subject_row[f"{mod}_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)
        paths = read_dicom_sequence(full_path)

        count = len(paths)
        if count > 0:
            # Naive Middle Slice
            idx = count // 2
            img_raw = read_image_file(paths[idx])
            # Independent normalization per channel (Cite solution_lesson_node_00023)
            img_proc = process_image(img_raw)
        else:
            img_proc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        final_channels.append(img_proc)

    # Stack into final tensor (Channels, H, W)
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

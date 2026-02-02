import os
import glob
import re
import random
import numpy as np
import pandas as pd
import torch
import cv2

# Attempt to import pydicom, but do not fail if missing as per package list constraints
# We prefer pydicom for header handling, but fallback to CV2/Filename parsing if needed.
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_instance_number(filename):
    """
    Extracts the integer instance number from the filename (e.g., 'Image-123.dcm' -> 123).
    Used for sorting slices spatially without reading DICOM headers.
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    # Fallback: try to find any number
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[-1])
    return 0


def read_dicom_image(path):
    """
    Reads a single DICOM file.
    Prioritizes pydicom if available, falls back to OpenCV.
    Returns a numpy array or None if failed.
    """
    # Method 1: pydicom (Preferred for medical correctness)
    if HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(path)
            return ds.pixel_array
        except Exception:
            pass

    # Method 2: OpenCV (Fallback)
    try:
        # cv2.IMREAD_UNCHANGED is crucial for 16-bit DICOMs
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def load_dicom_volume(folder_path):
    """
    Loads a volumetric MRI scan from a folder of DICOM slices.
    Slices are sorted by Instance Number (inferred from filename).

    Args:
        folder_path (str): Path to the directory containing .dcm files.

    Returns:
        np.ndarray: 3D numpy array (Depth, Height, Width) or None if empty.
    """
    if not os.path.exists(folder_path):
        return None

    files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
    if not files:
        return None

    # Sort files by instance number to ensure correct Z-ordering
    files.sort(key=lambda x: _extract_instance_number(os.path.basename(x)))

    slices = []
    for f in files:
        img = read_dicom_image(f)
        if img is not None:
            slices.append(img)

    if not slices:
        return None

    # Stack slices into a 3D volume (Depth, H, W)
    volume = np.stack(slices)
    return volume


def calculate_center_of_mass(volume):
    """
    Calculates the Center of Mass (CoM) along the Z-axis (depth) based on non-zero pixels.
    This anchors the sampling to the brain tissue, ignoring empty space.

    Args:
        volume (np.ndarray): 3D numpy array.

    Returns:
        int: The index of the slice corresponding to the center of mass.
    """
    if volume is None or volume.size == 0:
        return 0

    # Check if volume is empty (all zeros)
    if np.max(volume) == 0:
        return volume.shape[0] // 2

    # Get mass distribution along Z-axis (sum of non-zero pixels per slice)
    z_mass = np.sum(volume > 0, axis=(1, 2))  # Shape: (Depth,)

    total_mass = np.sum(z_mass)
    if total_mass == 0:
        return volume.shape[0] // 2

    # Weighted average of indices
    indices = np.arange(len(z_mass))
    center_float = np.sum(indices * z_mass) / total_mass

    return int(round(center_float))


def normalize_min_max(img):
    """
    Normalizes a 2D or 3D array to range [0, 1] using Min-Max scaling.
    Converts to float32.
    """
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def get_centroids(metadata_df, split_name="train", load_cached_data=True):
    """
    Computes or loads the Center of Mass (CoM) for each subject and modality.
    Implements caching using Parquet to avoid re-processing volumes.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing subject paths.
        split_name (str): 'train', 'val', or 'test' for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Original dataframe enriched with {Modality}_CoM columns.
    """
    Config.setup()  # Ensure directories exist
    cache_path = os.path.join(Config.WORK_DIR, f"centroids_{split_name}.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Verify it covers the requested IDs (subset check)
            if set(metadata_df["BraTS21ID"]).issubset(set(cached_df["BraTS21ID"])):
                print(f"Loaded centroids from cache: {cache_path}")
                # Merge cached columns back to the input metadata
                cols_to_use = ["BraTS21ID"] + [f"{m}_CoM" for m in Config.MODALITIES]
                # Ensure we don't duplicate columns if they already exist in metadata_df
                cols_to_merge = [
                    c
                    for c in cols_to_use
                    if c not in metadata_df.columns or c == "BraTS21ID"
                ]

                merged_df = metadata_df.merge(
                    cached_df[cols_to_merge], on="BraTS21ID", how="left"
                )
                return merged_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Computing centroids for {split_name} set ({len(metadata_df)} subjects)...")

    results = []

    # Iterate over subjects
    for idx, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]
        res = {"BraTS21ID": sid}

        for mod in Config.MODALITIES:
            # Construct full path. Metadata has relative paths.
            # e.g. row['flair_path'] -> 'train/00000/FLAIR'
            # Note: Config.MODALITIES are ["FLAIR", "T1wCE", "T2w"]
            # Metadata columns are "flair_path", "t1wce_path", "t2w_path"
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            volume = load_dicom_volume(full_path)
            if volume is not None:
                com = calculate_center_of_mass(volume)
            else:
                # Fallback if volume missing (rare/excluded cases)
                com = 0

            res[f"{mod}_CoM"] = com

        results.append(res)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(metadata_df)}")

    # 3. Save Cache
    centroids_df = pd.DataFrame(results)
    centroids_df.to_parquet(cache_path, index=False)
    print(f"Saved centroids to {cache_path}")

    # Merge back to original df
    merged_df = metadata_df.merge(centroids_df, on="BraTS21ID", how="left")
    return merged_df

import os
import re
import numpy as np
import pandas as pd
import cv2
import warnings

# Attempt to import pydicom, handle if missing (though expected to be present)
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import (
    INPUT_DIR,
    WORK_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MODALITIES,
    ROI_RELATIVE_DEPTHS,
)
from library.utils import get_logger

# Initialize Logger
logger = get_logger("roi_processing")


def natural_sort_key(s):
    """
    Key for natural sorting of strings containing numbers (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path):
    """
    Reads a DICOM file and returns the pixel array.
    Tries pydicom first, then cv2.
    Returns None if failure.
    """
    # Method 1: pydicom
    if pydicom is not None:
        try:
            dcm = pydicom.dcmread(path)
            return dcm.pixel_array
        except Exception:
            pass

    # Method 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def compute_roi_bounds(folder_rel_path):
    """
    Scans a directory of DICOM files to find the start and end indices
    of slices containing brain tissue (pixels > 0).

    Args:
        folder_rel_path: Relative path to the modality folder (e.g., 'train/00000/FLAIR')

    Returns:
        tuple: (start_index, end_index, sorted_filenames)
    """
    full_path = os.path.join(INPUT_DIR, folder_rel_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Modality directory not found: {full_path}")

    files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]

    if not files:
        raise ValueError(f"No DICOM files found in {full_path}")

    # Sort files numerically
    files.sort(key=natural_sort_key)

    start_idx = None
    end_idx = None

    # Iterate through files to find brain tissue
    # Note: This can be I/O intensive, but is required for strict ROI detection
    for idx, f in enumerate(files):
        file_path = os.path.join(full_path, f)
        img = read_dicom(file_path)

        if img is None:
            continue

        if np.max(img) > 0:
            if start_idx is None:
                start_idx = idx
            end_idx = idx

    if start_idx is None:
        # Strict Integrity: Raise error if no brain tissue found
        raise ValueError(
            f"No brain tissue (pixels > 0) found in any slice of {full_path}"
        )

    return start_idx, end_idx, files


def get_relative_indices(start, end, depths):
    """
    Calculates the indices corresponding to relative depths within the ROI [start, end].
    """
    roi_len = end - start + 1
    indices = []
    for d in depths:
        # Calculate relative offset
        offset = int(d * roi_len)
        # Determine absolute index
        idx = start + offset
        # Clamp to bounds
        idx = max(start, min(end, idx))
        indices.append(idx)
    return indices


def process_dataset(metadata_df, dataset_name):
    """
    Process a metadata dataframe to generate the cache of selected file paths.
    """
    logger.info(f"Processing {dataset_name} dataset ({len(metadata_df)} subjects)...")

    results = []

    for i, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]

        # Dictionary to store the selected paths for this subject
        subject_entry = {"BraTS21ID": subject_id}

        # If target exists, keep it
        if "MGMT_value" in row:
            subject_entry["MGMT_value"] = row["MGMT_value"]

        try:
            for mod in MODALITIES:
                # Get path from metadata (e.g., 'flair_path' column)
                col_name = f"{mod.lower()}_path"
                if col_name not in row:
                    raise KeyError(f"Column {col_name} missing in metadata")

                mod_path = row[col_name]

                # 1. Compute ROI
                start, end, sorted_files = compute_roi_bounds(mod_path)

                # 2. Get Indices for relative depths
                indices = get_relative_indices(start, end, ROI_RELATIVE_DEPTHS)

                # 3. Store selected file paths
                for depth_idx, file_idx in enumerate(indices):
                    # Construct the relative path to the specific file
                    # We store relative path to save space and remain flexible
                    file_name = sorted_files[file_idx]
                    full_file_rel_path = os.path.join(mod_path, file_name)

                    # Key format: MODALITY_DEPTH (e.g., FLAIR_0.4)
                    depth_key = str(ROI_RELATIVE_DEPTHS[depth_idx])
                    entry_key = f"{mod}_{depth_key}_path"
                    subject_entry[entry_key] = full_file_rel_path

            results.append(subject_entry)

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(metadata_df)} subjects")

        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            raise e

    return pd.DataFrame(results)


def generate_roi_cache(load_cached_data=True):
    """
    Generates or loads the ROI cache for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (df_train_cache, df_val_cache, df_test_cache)
    """
    cache_dir = WORK_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "roi_cache_train.parquet")
    val_cache_path = os.path.join(cache_dir, "roi_cache_val.parquet")
    test_cache_path = os.path.join(cache_dir, "roi_cache_test.parquet")

    # 1. Attempt Load
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            logger.info("Loading ROI cache from disk...")
            df_train = pd.read_parquet(train_cache_path)
            df_val = pd.read_parquet(val_cache_path)
            df_test = pd.read_parquet(test_cache_path)

            # Verify cache consistency (Cite debug_lesson_1)
            df_meta_train_check = pd.read_csv(TRAIN_METADATA_PATH)
            if len(df_train) == len(df_meta_train_check):
                logger.info(
                    f"Loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
                )
                return df_train, df_val, df_test
            else:
                logger.warning(
                    f"Cache mismatch: Found {len(df_train)} samples, expected {len(df_meta_train_check)}. "
                    "Cache is likely stale. Recomputing..."
                )
        else:
            logger.info("Cache files not found. Computing from scratch...")
    else:
        logger.info("Forcing re-computation of ROI cache...")

    # 2. Load Metadata
    df_meta_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_meta_val = pd.read_csv(VAL_METADATA_PATH)
    df_meta_test = pd.read_csv(TEST_METADATA_PATH)

    # 3. Process Datasets
    df_train = process_dataset(df_meta_train, "Train")
    df_val = process_dataset(df_meta_val, "Validation")
    df_test = process_dataset(df_meta_test, "Test")

    # 4. Integrity Checks
    logger.info("Running Integrity Checks...")

    # Check lengths
    assert len(df_train) == len(df_meta_train), "Train cache length mismatch"
    assert len(df_val) == len(df_meta_val), "Val cache length mismatch"
    assert len(df_test) == len(df_meta_test), "Test cache length mismatch"

    # Check for NaNs
    if df_train.isnull().values.any():
        raise ValueError("NaN values found in Train cache")
    if df_val.isnull().values.any():
        raise ValueError("NaN values found in Val cache")
    if df_test.isnull().values.any():
        raise ValueError("NaN values found in Test cache")

    # 5. Save Cache
    logger.info(f"Saving cache to {cache_dir}...")
    df_train.to_parquet(train_cache_path)
    df_val.to_parquet(val_cache_path)
    df_test.to_parquet(test_cache_path)

    logger.info("ROI Cache generation complete.")

    return df_train, df_val, df_test

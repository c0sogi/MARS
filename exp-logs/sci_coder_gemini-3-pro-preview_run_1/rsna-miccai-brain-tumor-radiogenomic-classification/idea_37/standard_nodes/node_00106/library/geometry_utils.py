import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from library.config import Config


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm comes before Image-10.dcm).
    Splits string into text and numeric parts.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_image(path):
    """
    Attempts to read a DICOM file's pixel array.
    Tries pydicom first (standard), falls back to OpenCV (cv2).
    Returns a numpy array or None if reading fails.
    """
    # Attempt 1: pydicom
    # Although not explicitly in the installed list, EDA logs showed success with it.
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        return dcm.pixel_array
    except (ImportError, Exception):
        pass

    # Attempt 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def analyze_modality_geometry(file_paths):
    """
    Calculates the Z-axis Center of Mass (CoM) and Brain ROI Depth
    based on the presence of non-zero pixels (brain tissue).

    Args:
        file_paths: List of sorted file paths for a single modality.

    Returns:
        tuple: (com_index, depth, num_files)
    """
    total_mass = 0.0
    weighted_sum_z = 0.0
    min_z = None
    max_z = None

    num_files = len(file_paths)
    if num_files == 0:
        return 0.0, 0, 0

    # Iterate through slices to build the spatial distribution
    for z, filepath in enumerate(file_paths):
        img = read_dicom_image(filepath)

        if img is None:
            continue

        # "Mass" is defined as the count of non-zero pixels (tissue)
        # This is robust to intensity variations across scanners
        mass = np.count_nonzero(img)

        if mass > 0:
            total_mass += mass
            weighted_sum_z += z * mass

            if min_z is None:
                min_z = z
            max_z = z

    # Handle cases with no signal (blank scans)
    if total_mass == 0 or min_z is None:
        # Fallback to geometric center
        com_index = num_files / 2.0
        depth = 0
    else:
        # Calculate Center of Mass
        com_index = weighted_sum_z / total_mass
        # Calculate Depth of the Brain ROI
        depth = max_z - min_z

    return com_index, depth, num_files


def get_target_indices(com_index, depth, num_files, offsets):
    """
    Calculates the specific integer file indices for the requested relative offsets.

    Args:
        com_index: The calculated Center of Mass index (float).
        depth: The depth of the brain ROI (int).
        num_files: Total number of files in the stack.
        offsets: List of float offsets (e.g., [-0.1, 0.0, 0.1]).

    Returns:
        list: List of integer indices clamped to [0, num_files-1].
    """
    indices = []
    for offset in offsets:
        # Calculate target index: CoM + (Offset * ROI_Depth)
        target = com_index + (offset * depth)

        # Round to nearest integer
        idx = int(np.round(target))

        # Clamp to valid range
        idx = max(0, min(idx, num_files - 1))
        indices.append(idx)
    return indices


def process_subject_geometry(metadata_df, load_cached_data=True):
    """
    Main entry point to process geometry for a dataset.
    Calculates slice indices for the ARVS network input.

    Args:
        metadata_df: DataFrame containing subject IDs and paths.
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        DataFrame containing BraTS21ID and calculated indices for each modality/offset.
    """
    # Construct a cache filename based on the dataset size to distinguish train/val/test
    # (or simply use a consistent hash if needed, but size is a good proxy here)
    cache_filename = f"geometry_cache_{len(metadata_df)}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading geometry cache from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from scratch
    print(
        f"Processing geometry for {len(metadata_df)} subjects. This may take a while..."
    )
    results = []

    for i, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]
        subject_data = {"BraTS21ID": subject_id}

        for mod in Config.MODALITIES:
            # Construct full path to the modality directory
            # Metadata contains relative paths, e.g., 'train/00000/FLAIR'
            rel_path = row[f"{mod}_path"]
            full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Get sorted file list
            if os.path.exists(full_dir_path):
                files = glob.glob(os.path.join(full_dir_path, "*.dcm"))
                files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
            else:
                files = []

            # Analyze geometry (CoM and Depth)
            com, depth, num_files = analyze_modality_geometry(files)

            # Calculate indices for the configured offsets
            indices = get_target_indices(com, depth, num_files, Config.RELATIVE_OFFSETS)

            # Store indices in the result dictionary
            # Column naming convention: {mod}_idx_{offset_position}
            for k, idx in enumerate(indices):
                col_name = f"{mod}_idx_{k}"
                subject_data[col_name] = idx

        results.append(subject_data)

        # Simple progress logging
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(metadata_df)} subjects...")

    # 3. Save to cache
    geometry_df = pd.DataFrame(results)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    geometry_df.to_parquet(cache_path)
    print(f"Saved geometry cache to {cache_path}")

    return geometry_df

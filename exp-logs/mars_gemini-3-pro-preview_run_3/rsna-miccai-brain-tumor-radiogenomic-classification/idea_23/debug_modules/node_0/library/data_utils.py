import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    NUM_SLICES_PER_MODALITY,
    MODALITIES,
    SEED,
)


def read_dicom(path):
    """
    Robustly reads a DICOM file and returns the pixel array.
    Returns a zero array if reading fails.
    """
    try:
        full_path = os.path.join(INPUT_DIR, path)
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array
        return img
    except Exception as e:
        # Return a zero placeholder of expected size if read fails
        # We assume a default size, will be resized later anyway
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def get_instance_number(path):
    """
    Extracts the Instance Number (0020,0013) from a DICOM file.
    Returns -1 if the tag is missing or file is unreadable.
    """
    try:
        full_path = os.path.join(INPUT_DIR, path)
        # stop_before_pixels=True speeds up reading significantly
        dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
        return int(dcm.InstanceNumber)
    except Exception:
        return -1


def sort_paths_by_instance(paths):
    """
    Sorts a list of file paths based on the DICOM Instance Number.
    """
    # Create a list of (instance_number, path) tuples
    path_instances = []
    for p in paths:
        inst = get_instance_number(p)
        path_instances.append((inst, p))

    # Sort by instance number
    path_instances.sort(key=lambda x: x[0])

    # Return just the paths
    return [p for _, p in path_instances]


def uniform_sample(paths):
    """
    Selects 32 slices uniformly distributed across the 10%-90% depth range.
    """
    n_files = len(paths)
    if n_files == 0:
        return []

    # Define depth range (10% to 90%)
    start_idx = int(n_files * 0.10)
    end_idx = int(n_files * 0.90)

    # Ensure we have a valid range
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = n_files

    # Generate indices
    # We want exactly NUM_SLICES_PER_MODALITY indices
    indices = np.linspace(start_idx, end_idx - 1, NUM_SLICES_PER_MODALITY, dtype=int)

    # Clip indices just in case
    indices = np.clip(indices, 0, n_files - 1)

    # Select paths
    sampled_paths = [paths[i] for i in indices]
    return sampled_paths


def global_volumetric_normalize(volume):
    """
    Normalizes the volume to [0, 1] based on global min and max.
    Volume shape: (D, H, W)
    """
    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def load_patient_volume(row):
    """
    Orchestrates loading, sorting, sampling, and stacking for a single patient.
    Returns a tensor of shape (128, 224, 224).
    """
    all_slices = []

    # Iterate through modalities in strict order: FLAIR, T1w, T1wCE, T2w
    for mod in MODALITIES:
        col_name = f"{mod.lower()}_paths"
        paths = row[col_name]

        # Explicit check for missing data
        if paths is None or len(paths) == 0:
            # Handle missing modality by creating empty slices (zeros)
            # This prevents silent failure/crash
            mod_volume = np.zeros(
                (NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32
            )
            all_slices.append(mod_volume)
            continue

        # 1. Sort by Instance Number (Spatial Coherence)
        sorted_paths = sort_paths_by_instance(paths)

        # 2. Uniform Sampling (High-Density)
        sampled_paths = uniform_sample(sorted_paths)

        # 3. Load and Resize Images
        mod_slices = []
        for p in sampled_paths:
            img = read_dicom(p)

            # Resize if necessary
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
                )

            mod_slices.append(img)

        # Stack slices for this modality -> (32, 224, 224)
        mod_volume = np.array(mod_slices, dtype=np.float32)
        all_slices.append(mod_volume)

    # Concatenate all modalities along the depth/channel dimension
    # Result shape: (4 * 32, 224, 224) = (128, 224, 224)
    full_volume = np.concatenate(all_slices, axis=0)

    # 4. Global Volumetric Normalization
    full_volume = global_volumetric_normalize(full_volume)

    return full_volume


def load_dataset(metadata_df, cache_name, load_cached_data=True):
    """
    Loads the dataset, using caching to save time on subsequent runs.

    Args:
        metadata_df: DataFrame containing patient metadata and file paths.
        cache_name: String identifier for the cache (e.g., 'train', 'val', 'test').
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        X: Numpy array of inputs (N, 128, 224, 224)
        y: Numpy array of targets (N,) or None for test set
        ids: Numpy array of BraTS21IDs (N,)
    """
    # Define cache file paths
    cache_X_path = os.path.join(WORKING_DIR, f"cached_{cache_name}_X.npy")
    cache_y_path = os.path.join(WORKING_DIR, f"cached_{cache_name}_y.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"cached_{cache_name}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            print(f"Loading {cache_name} data from cache...")
            X = np.load(cache_X_path)
            ids = np.load(cache_ids_path)

            y = None
            if os.path.exists(cache_y_path):
                y = np.load(cache_y_path)

            return X, y, ids
        else:
            print(f"Cache not found for {cache_name}. Processing from scratch...")
    else:
        print(f"Ignoring cache for {cache_name}. Processing from scratch...")

    # 2. Process data from scratch
    X_list = []
    y_list = []
    ids_list = []

    print(f"Processing {len(metadata_df)} subjects for {cache_name} set...")

    for idx, row in metadata_df.iterrows():
        # Load volume
        volume = load_patient_volume(row)
        X_list.append(volume)

        # Store ID
        ids_list.append(row["BraTS21ID"])

        # Store Target if available
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to cache
    print(f"Saving {cache_name} data to cache...")
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids

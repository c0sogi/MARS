import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import cv2
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    IMAGE_SIZE,
    SLICES_PER_MODALITY,
    TOTAL_CHANNELS,
    SEED,
)


def extract_slice_number(path):
    """
    Extracts the integer slice number from the filename (e.g., Image-10.dcm -> 10).
    """
    basename = os.path.basename(path)
    match = re.search(r"Image-(\d+)\.dcm", basename)
    if match:
        return int(match.group(1))
    return 0


def get_sorted_image_files(file_paths):
    """
    Sorts file paths based on the integer slice number in the filename.
    """
    return sorted(file_paths, key=extract_slice_number)


def process_modality_slice(path):
    """
    Loads a DICOM slice, converts to float32, and resizes it.
    """
    full_path = os.path.join(INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(
                img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
            )

        return img
    except Exception:
        # Return zero slice on failure
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


def load_dicom_volume(paths):
    """
    Loads, sorts, samples, and normalizes a volume for a single modality.
    Applies View-Adaptive Per-Modality Normalization.
    """
    if not paths or len(paths) == 0:
        return np.zeros((SLICES_PER_MODALITY, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # 1. External Integer Sorting
    sorted_paths = get_sorted_image_files(paths)

    # 2. Uniform Sampling
    total_slices = len(sorted_paths)
    if total_slices < SLICES_PER_MODALITY:
        # If fewer slices than desired, sample with replacement or just linspace over available
        indices = np.linspace(0, total_slices - 1, SLICES_PER_MODALITY).astype(int)
    else:
        indices = np.linspace(0, total_slices - 1, SLICES_PER_MODALITY).astype(int)

    selected_paths = [sorted_paths[i] for i in indices]

    # 3. Load Slices
    volume_slices = [process_modality_slice(p) for p in selected_paths]
    volume = np.array(volume_slices)  # Shape: (16, 320, 320)

    # 4. View-Adaptive Per-Modality Normalization
    # Calculate min/max only within the sampled subset
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 MGMT prediction.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        tensor_img = torch.tensor(img, dtype=torch.float32)

        if self.y is not None:
            label = self.y[idx]
            return tensor_img, torch.tensor(label, dtype=torch.float32)

        return tensor_img


def load_processed_data(split_name, load_cached_data=True, max_samples=None):
    """
    Loads data for a specific split ('train', 'val', 'test').
    Implements caching using .npy files in WORKING_DIR.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_X_path = os.path.join(WORKING_DIR, f"cached_{split_name}_X.npy")
    cache_y_path = os.path.join(WORKING_DIR, f"cached_{split_name}_y.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"cached_{split_name}_ids.npy")

    # Load metadata to determine expected size
    meta_path = os.path.join(METADATA_DIR, f"{split_name}.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)
    expected_len = len(df)
    if max_samples is not None:
        expected_len = min(expected_len, max_samples)

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_X_path)
        and os.path.exists(cache_ids_path)
    ):
        # For non-test sets, check if y exists
        if split_name == "test" or os.path.exists(cache_y_path):
            X = np.load(cache_X_path)

            # Cite debug_lesson_1: Validate Cached Data Against Current Configuration
            # Verify if cached data meets the sample count requirements
            if len(X) == expected_len or (
                max_samples is not None and len(X) > expected_len
            ):
                print(f"Loading cached {split_name} data from {WORKING_DIR}...")
                ids = np.load(cache_ids_path, allow_pickle=True)
                y = None
                if split_name != "test":
                    y = np.load(cache_y_path)

                if max_samples is not None:
                    return (
                        X[:max_samples],
                        (y[:max_samples] if y is not None else None),
                        ids[:max_samples],
                    )
                return X, y, ids
            else:
                print(
                    f"Cache mismatch for {split_name}: Expected {expected_len} samples, found {len(X)}. Regenerating..."
                )

    # Process from scratch
    print(f"Processing {split_name} data from scratch...")

    if max_samples is not None:
        df = df.iloc[:max_samples]

    X_list = []
    y_list = []
    ids_list = []

    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for idx, row in df.iterrows():
        patient_chunks = []
        for mod in modalities:
            col_name = f"{mod}_paths"
            paths = row[col_name]

            # Handle None/NaN/Empty
            if paths is None:
                paths = []
            if isinstance(paths, np.ndarray):
                paths = paths.tolist()

            mod_vol = load_dicom_volume(paths)
            patient_chunks.append(mod_vol)

        # Stack Modality Blocks: (4, 16, 320, 320) -> (64, 320, 320)
        full_volume = np.concatenate(patient_chunks, axis=0)

        X_list.append(full_volume)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)

    y = None
    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y_path, y)

    return X, y, ids

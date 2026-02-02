import os
import re
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    NUM_SLICES_PER_MODALITY,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    MODALITIES,
)
from library.utils import seed_everything


def extract_slice_number(path):
    """
    Extracts the integer slice number from a DICOM filename.
    Expected format: *-{number}.dcm
    """
    match = re.search(r"Image-(\d+)\.dcm", path)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_slice(path):
    """
    Reads a DICOM file and returns the pixel array.
    Returns None if reading fails.
    """
    full_path = os.path.join(INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except Exception:
        return None


def process_modality_files(file_paths):
    """
    Process a list of file paths for a single modality:
    1. Sort by slice number.
    2. Uniformly sample NUM_SLICES_PER_MODALITY slices from 10%-90% range.
    3. Resize to IMG_SIZE x IMG_SIZE.
    4. Apply View-Adaptive Normalization (min-max on the selected stack).

    Returns:
        np.ndarray: Shape (NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE)
    """
    # 1. Sort
    # Filter out paths that don't exist or are invalid (though metadata should be clean)
    valid_paths = [p for p in file_paths if p]

    if len(valid_paths) == 0:
        return np.zeros((NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    sorted_paths = sorted(valid_paths, key=extract_slice_number)

    # 2. Uniform Sampling (10% to 90%)
    num_files = len(sorted_paths)
    if num_files < NUM_SLICES_PER_MODALITY:
        # If fewer slices than required, take all and pad (or repeat)
        # Strategy: Linear interpolation indices, effectively repeating some slices
        indices = np.linspace(0, num_files - 1, NUM_SLICES_PER_MODALITY)
    else:
        # Crop top/bottom 10% to avoid artifacts, then sample
        start_idx = int(num_files * 0.1)
        end_idx = int(num_files * 0.9)
        # Ensure we have a valid range
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = num_files

        indices = np.linspace(start_idx, end_idx - 1, NUM_SLICES_PER_MODALITY)

    indices = np.round(indices).astype(int)
    selected_paths = [sorted_paths[i] for i in indices]

    # 3. Load and Resize
    slice_list = []
    for p in selected_paths:
        img = load_dicom_slice(p)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            # Resize
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )
        slice_list.append(img)

    stack = np.array(slice_list)  # (16, 320, 320)

    # 4. View-Adaptive Normalization
    # Compute min/max only on this specific stack of 16 slices
    min_val = np.min(stack)
    max_val = np.max(stack)

    if max_val - min_val > 0:
        stack = (stack - min_val) / (max_val - min_val)
    else:
        stack = np.zeros_like(stack)

    return stack


def process_patient(row):
    """
    Process all modalities for a single patient.
    Returns:
        X: (64, 320, 320) float32 tensor
        y: target value (or -1 if test)
    """
    # Modality order: FLAIR, T1w, T1wCE, T2w
    chunks = []

    # Map modality names to dataframe columns
    # Dataframe columns are like 'flair_paths', 't1w_paths', etc.
    # Config MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for mod in MODALITIES:
        col_name = f"{mod.lower()}_paths"
        paths = row[col_name]
        # Handle potential None or NaN
        if not isinstance(paths, list):
            paths = []

        mod_stack = process_modality_files(paths)
        chunks.append(mod_stack)

    # Stack along channel dimension (0)
    # Each chunk is (16, 320, 320) -> Result (64, 320, 320)
    X = np.concatenate(chunks, axis=0).astype(np.float32)

    if "MGMT_value" in row:
        y = float(row["MGMT_value"])
    else:
        y = -1.0

    return X, y


def get_dataset_arrays(meta_path, cache_prefix, load_cached_data=True):
    """
    Loads dataset arrays from cache or processes them from scratch.

    Args:
        meta_path: Path to parquet metadata file.
        cache_prefix: Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data: Boolean flag to use cache.

    Returns:
        X: numpy array of shape (N, 64, 320, 320)
        y: numpy array of shape (N,)
        ids: numpy array of BraTS21IDs
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_X_path = os.path.join(CACHE_DIR, f"cached_{cache_prefix}_X.npy")
    cache_y_path = os.path.join(CACHE_DIR, f"cached_{cache_prefix}_y.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"cached_{cache_prefix}_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(cache_X_path)
            and os.path.exists(cache_y_path)
            and os.path.exists(cache_ids_path)
        ):
            print(f"Loading {cache_prefix} data from cache...")
            try:
                X = np.load(cache_X_path)
                y = np.load(cache_y_path)
                ids = np.load(cache_ids_path, allow_pickle=True)
                return X, y, ids
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

    # 2. Process from Scratch
    print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_parquet(meta_path)

    X_list = []
    y_list = []
    ids_list = []

    # Iterate over dataframe
    for idx, row in df.iterrows():
        X_pat, y_pat = process_patient(row)
        X_list.append(X_pat)
        y_list.append(y_pat)
        ids_list.append(str(row["BraTS21ID"]))

        # Optional: Print progress every 50 samples
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)

    # 3. Save to Cache
    print(f"Saving {cache_prefix} data to cache at {CACHE_DIR}...")
    np.save(cache_X_path, X)
    np.save(cache_y_path, y)
    np.save(cache_ids_path, ids)

    return X, y, ids


class MGMTDataset(Dataset):
    """
    Simple wrapper for pre-loaded numpy arrays.
    """

    def __init__(self, X, y, ids, is_test=False):
        self.X = X
        self.y = y
        self.ids = ids
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (64, 320, 320)
        # PyTorch expects float32 tensors
        img = torch.from_numpy(self.X[idx])

        if self.is_test:
            # For test, return ID as well for submission
            return img, self.ids[idx]
        else:
            # For train/val, return target
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, target


def get_dataloaders(
    train_meta_path, val_meta_path, test_meta_path, load_cached_data=True
):
    """
    Generates DataLoaders for train, val, and test sets.
    Handles caching and data processing.
    """
    seed_everything(SEED)

    # --- Train ---
    if os.path.exists(train_meta_path):
        X_train, y_train, ids_train = get_dataset_arrays(
            train_meta_path, "train", load_cached_data
        )
        train_dataset = MGMTDataset(X_train, y_train, ids_train, is_test=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        train_loader = None
        print(f"Warning: Train metadata not found at {train_meta_path}")

    # --- Val ---
    if os.path.exists(val_meta_path):
        X_val, y_val, ids_val = get_dataset_arrays(
            val_meta_path, "val", load_cached_data
        )
        val_dataset = MGMTDataset(X_val, y_val, ids_val, is_test=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        val_loader = None
        print(f"Warning: Val metadata not found at {val_meta_path}")

    # --- Test ---
    if os.path.exists(test_meta_path):
        X_test, y_test, ids_test = get_dataset_arrays(
            test_meta_path, "test", load_cached_data
        )
        test_dataset = MGMTDataset(X_test, y_test, ids_test, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        test_loader = None
        print(f"Warning: Test metadata not found at {test_meta_path}")

    return train_loader, val_loader, test_loader

import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import load_metadata, seed_everything

# ==========================================
# Constants & Configuration
# ==========================================
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_11"
IMG_SIZE = 256
NUM_SLICES = 32
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]


# ==========================================
# Helper Functions
# ==========================================
def load_dicom_slice(path):
    """
    Loads a single DICOM file and returns the pixel array.
    Returns a zero array if loading fails.
    """
    try:
        full_path = os.path.join(INPUT_DIR, path)
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def process_modality(paths):
    """
    Processes a list of file paths for a single modality:
    1. Samples 32 slices uniformly from the 10%-90% depth range.
    2. Loads and resizes images to 256x256.
    3. Performs Global Volumetric Normalization (min-max scaling).

    Returns:
        np.ndarray: Shape (32, 256, 256)
    """
    if not paths:
        return np.zeros((NUM_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    n_files = len(paths)

    # 1. High-Density Uniform Sampling (10% - 90%)
    start = int(n_files * 0.1)
    end = int(n_files * 0.9)

    # Handle edge cases with few slices
    if end <= start:
        start = 0
        end = n_files

    # Generate indices
    if n_files < NUM_SLICES:
        # If fewer files than target, interpolate to stretch them out
        indices = np.linspace(0, n_files - 1, NUM_SLICES).astype(int)
    else:
        # Uniformly sample within the range
        # Ensure we don't go out of bounds
        limit = max(start + 1, end)
        indices = np.linspace(start, limit - 1, NUM_SLICES).astype(int)

    selected_paths = [paths[i] for i in indices]

    # 2. Load and Resize
    volume = []
    for p in selected_paths:
        img = load_dicom_slice(p)

        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        volume.append(img)

    volume = np.array(volume, dtype=np.float32)  # Shape: (32, 256, 256)

    # 3. Global Volumetric Normalization
    # We normalize based on the min/max of the sampled volume to preserve contrast
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def prepare_data(split, load_cached_data=True, debug_limit=None):
    """
    Loads metadata, processes raw DICOMs into tensors, and caches results.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_limit (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (X, y, ids)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_X = os.path.join(CACHE_DIR, f"X_{split}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_{split}.npy")
    cache_ids = os.path.join(CACHE_DIR, f"ids_{split}.npy")

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_X) and os.path.exists(cache_ids):
            # Check y existence only if not test (test might not have y)
            if split == "test" or os.path.exists(cache_y):
                print(f"Loading cached {split} data from {CACHE_DIR}...")
                X = np.load(cache_X)
                ids = np.load(cache_ids)
                y = np.load(cache_y) if os.path.exists(cache_y) else None

                # If debug_limit is set, slice the cached data
                if debug_limit:
                    X = X[:debug_limit]
                    ids = ids[:debug_limit]
                    if y is not None:
                        y = y[:debug_limit]
                return X, y, ids

    # Process from scratch
    print(f"Processing {split} data from scratch...")
    df = load_metadata(split)

    if debug_limit:
        df = df.head(debug_limit)

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process each modality and stack
        # Modality Order: FLAIR, T1w, T1wCE, T2w
        mod_volumes = []
        for mod in MODALITIES:
            paths = row.get(f"{mod}_paths", [])
            vol = process_modality(paths)  # (32, 256, 256)
            mod_volumes.append(vol)

        # Stack along channel dimension (depth)
        # Result: (4 * 32, 256, 256) = (128, 256, 256)
        full_volume = np.concatenate(mod_volumes, axis=0)

        X_list.append(full_volume)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # Save to cache
    print(f"Saving {split} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


# ==========================================
# Dataset Class
# ==========================================
class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None, is_test=False):
        self.X = X
        self.y = y
        self.ids = ids
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Input shape: (128, 256, 256)
        img = self.X[idx]
        img_tensor = torch.from_numpy(img).float()

        if self.is_test:
            patient_id = self.ids[idx]
            return img_tensor, patient_id
        else:
            label = self.y[idx]
            return img_tensor, torch.tensor(label, dtype=torch.float)


# ==========================================
# Data Loader Factory
# ==========================================
def get_dataloaders(batch_size=8, load_cached_data=True, debug_limit=None):
    """
    Generates DataLoaders for train, val, and test splits.

    Args:
        batch_size (int): Batch size for loaders.
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug_limit (int): Optional limit for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(42)

    # Prepare Data
    X_train, y_train, ids_train = prepare_data("train", load_cached_data, debug_limit)
    X_val, y_val, ids_val = prepare_data("val", load_cached_data, debug_limit)
    X_test, y_test, ids_test = prepare_data("test", load_cached_data, debug_limit)

    # Create Datasets
    train_dataset = BraTSDataset(X_train, y_train, ids_train, is_test=False)
    val_dataset = BraTSDataset(X_val, y_val, ids_val, is_test=False)
    test_dataset = BraTSDataset(X_test, y=None, ids=ids_test, is_test=True)

    # Create Loaders
    # num_workers=4 is safe for 12 vCPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

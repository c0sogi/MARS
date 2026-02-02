import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    SLICES_PER_MODALITY,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    NUM_MODALITIES,
)

# ==========================================
# DICOM Processing Utilities
# ==========================================


def load_dicom_volume(file_paths):
    """
    Reads a list of DICOM files, sorts them by Instance Number, and returns a 3D numpy array.
    """
    slices = []
    # Explicit check for None or empty to handle potential numpy arrays safely
    if file_paths is None or len(file_paths) == 0:
        return np.array([])

    for path in file_paths:
        full_path = os.path.join(INPUT_DIR, path)
        if os.path.exists(full_path):
            try:
                dcm = pydicom.dcmread(full_path)
                # Extract pixel array and Instance Number
                if hasattr(dcm, "pixel_array") and hasattr(dcm, "InstanceNumber"):
                    slices.append((int(dcm.InstanceNumber), dcm.pixel_array))
            except Exception:
                continue

    # Sort by Instance Number to ensure spatial consistency
    slices.sort(key=lambda x: x[0])

    if not slices:
        return np.array([])

    # Stack images: (Depth, Height, Width)
    volume = np.stack([s[1] for s in slices])
    return volume


def process_modality(file_paths):
    """
    Loads, samples, resizes, and normalizes a single modality volume.
    Returns a tensor of shape (SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE).
    """
    volume = load_dicom_volume(file_paths)

    # Initialize output container
    processed_volume = np.zeros(
        (SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32
    )

    if volume.ndim != 3 or volume.shape[0] == 0:
        return processed_volume

    depth = volume.shape[0]

    # High-Density Uniform Sampling (10% to 90%)
    if depth >= SLICES_PER_MODALITY:
        start_idx = int(depth * 0.1)
        end_idx = int(depth * 0.9)
        # Ensure start < end
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = depth

        indices = np.linspace(start_idx, end_idx - 1, SLICES_PER_MODALITY, dtype=int)
    else:
        # If fewer slices than required, take all and pad later (already init with zeros)
        indices = np.arange(depth)

    # Extract sampled slices
    sampled_slices = []
    for idx in indices:
        if idx < depth:
            slc = volume[idx]
            # Resize
            if slc.shape[0] != IMG_SIZE or slc.shape[1] != IMG_SIZE:
                slc = cv2.resize(
                    slc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )
            sampled_slices.append(slc)

    if not sampled_slices:
        return processed_volume

    sampled_volume = np.stack(sampled_slices)  # (N, 224, 224)

    # Global Volumetric Normalization (Min-Max)
    v_min = np.min(sampled_volume)
    v_max = np.max(sampled_volume)

    if v_max - v_min > 0:
        sampled_volume = (sampled_volume - v_min) / (v_max - v_min)
    else:
        sampled_volume = np.zeros_like(sampled_volume)

    # Fill into the fixed size container
    # If we had fewer slices, they go at the beginning (or center, but beginning is fine for consistency)
    valid_count = min(len(sampled_volume), SLICES_PER_MODALITY)
    processed_volume[:valid_count] = sampled_volume[:valid_count]

    return processed_volume


def process_patient(row):
    """
    Processes all 4 modalities for a single patient and stacks them.
    Returns: (128, 224, 224) numpy array.
    """
    # Order: FLAIR, T1w, T1wCE, T2w
    modalities = ["flair_paths", "t1w_paths", "t1wce_paths", "t2w_paths"]
    chunks = []

    for mod_col in modalities:
        paths = row[mod_col] if row[mod_col] is not None else []
        mod_vol = process_modality(paths)  # (32, 224, 224)
        chunks.append(mod_vol)

    # Stack along channel dimension: (4*32, 224, 224) -> (128, 224, 224)
    full_volume = np.concatenate(chunks, axis=0)
    return full_volume


# ==========================================
# Caching Logic
# ==========================================


def load_or_create_dataset(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads dataset from cache if available, otherwise processes from scratch and caches.
    Returns: X (numpy array), y (numpy array), ids (numpy array)
    """
    cache_X = os.path.join(CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_ids = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    expected_count = len(df)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        cached_ids = np.load(cache_ids)
        if len(cached_ids) == expected_count:
            print(f"Loading {cache_prefix} data from cache...")
            X = np.load(
                cache_X, mmap_mode="r"
            )  # Use mmap to save RAM if needed, though we load to RAM later
            ids = cached_ids

            # y might not exist for test set
            if os.path.exists(cache_y):
                y = np.load(cache_y)
            else:
                y = None

            return np.array(X), y, ids
        else:
            print(
                f"Cache mismatch for {cache_prefix}: Expected {expected_count} samples, found {len(cached_ids)}. Regenerating..."
            )

    # 2. Process from Scratch
    print(f"Processing {cache_prefix} data from scratch...")
    # df is already loaded

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process volume
        vol = process_patient(row)
        X_list.append(vol)
        ids_list.append(row["BraTS21ID"])

        # Process label if available
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.stack(X_list).astype(np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to Cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


# ==========================================
# PyTorch Dataset
# ==========================================


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None, transform=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (128, 224, 224)
        img = self.X[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        # Return dict
        sample = {
            "image": img_tensor,
            "BraTS21ID": self.ids[idx] if self.ids is not None else "",
        }

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            sample["target"] = label

        return sample


# ==========================================
# Main Interface
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Main function to get training and validation dataloaders.
    Handles caching and dataset creation.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------------------------
    # Load Training Data
    # ---------------------------
    X_train, y_train, ids_train = load_or_create_dataset(
        TRAIN_META_PATH, "cached_train", load_cached_data
    )

    # ---------------------------
    # Load Validation Data
    # ---------------------------
    X_val, y_val, ids_val = load_or_create_dataset(
        VAL_META_PATH, "cached_val", load_cached_data
    )

    # ---------------------------
    # Create Datasets
    # ---------------------------
    train_dataset = BraTSDataset(X_train, y_train, ids_train)
    val_dataset = BraTSDataset(X_val, y_val, ids_val)

    # ---------------------------
    # Create DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Returns the test dataloader.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    X_test, y_test, ids_test = load_or_create_dataset(
        TEST_META_PATH, "cached_test", load_cached_data
    )

    test_dataset = BraTSDataset(X_test, y=None, ids=ids_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader

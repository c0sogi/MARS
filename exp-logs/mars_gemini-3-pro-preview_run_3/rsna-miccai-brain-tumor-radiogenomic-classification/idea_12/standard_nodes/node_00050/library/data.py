import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_TRAIN_X,
    CACHE_TRAIN_Y,
    CACHE_VAL_X,
    CACHE_VAL_Y,
    CACHE_TEST_X,
    CACHE_TEST_IDS,
    NUM_SLICES,
    IMG_SIZE,
    MODALITIES,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma classification.
    """

    def __init__(self, X, y=None, ids=None, transform=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (Channels, Height, Width) -> (128, 256, 256)
        img = self.X[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label

        if self.ids is not None:
            return img_tensor, self.ids[idx]

        return img_tensor


def load_dicom_slice(path, img_size=IMG_SIZE):
    """
    Reads a DICOM file and resizes it.
    """
    full_path = os.path.join(INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array

        # Handle resizing
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception as e:
        # Return zero slice in case of error
        return np.zeros((img_size, img_size), dtype=np.float32)


def process_patient_volume(row, img_size=IMG_SIZE, num_slices=NUM_SLICES):
    """
    Loads, samples, normalizes, and stacks MRI volumes for a single patient.
    Returns a tensor of shape (num_slices * 4, img_size, img_size).
    """
    # Shape: (Num_Slices, Num_Modalities, H, W)
    # This will later be reshaped to (Num_Slices * Num_Modalities, H, W)
    # to achieve the interleaved ordering [S0_M0, S0_M1, S0_M2, S0_M3, S1_M0...]
    patient_volume = np.zeros(
        (num_slices, len(MODALITIES), img_size, img_size), dtype=np.float32
    )

    for m_idx, mod in enumerate(MODALITIES):
        col_name = f"{mod.lower()}_paths"
        paths = row[col_name]

        if paths is None or len(paths) == 0:
            continue

        # 1. High-Density Uniform Sampling (10% - 90%)
        total_slices = len(paths)
        start_idx = int(total_slices * 0.1)
        end_idx = int(total_slices * 0.9)

        # Ensure valid range
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = total_slices

        # Select indices uniformly
        indices = np.linspace(
            start_idx, max(start_idx, end_idx - 1), num_slices, dtype=int
        )
        selected_paths = [paths[i] for i in indices]

        # Load slices for this modality
        mod_slices = []
        for p in selected_paths:
            mod_slices.append(load_dicom_slice(p, img_size))

        mod_volume = np.array(mod_slices, dtype=np.float32)

        # 2. Global Volumetric Normalization (per modality)
        min_val = np.min(mod_volume)
        max_val = np.max(mod_volume)

        if max_val - min_val > 0:
            mod_volume = (mod_volume - min_val) / (max_val - min_val)
        else:
            mod_volume = np.zeros_like(mod_volume)

        # Assign to patient volume
        patient_volume[:, m_idx, :, :] = mod_volume

    # 3. Interleaved Stacking
    # Reshape from (32, 4, 256, 256) to (128, 256, 256)
    # This results in channels: S0_M0, S0_M1, S0_M2, S0_M3, S1_M0...
    final_volume = patient_volume.reshape(-1, img_size, img_size)

    return final_volume


def generate_dataset(
    metadata_path,
    cache_x_path,
    cache_y_path=None,
    cache_ids_path=None,
    load_cached_data=True,
):
    """
    Generates or loads the dataset from cache.
    """
    # Check if cache exists and we want to load it
    has_cache = os.path.exists(cache_x_path)
    if cache_y_path:
        has_cache = has_cache and os.path.exists(cache_y_path)
    if cache_ids_path:
        has_cache = has_cache and os.path.exists(cache_ids_path)

    if load_cached_data and has_cache:
        print(f"Loading cached data from {os.path.dirname(cache_x_path)}...")
        X = np.load(cache_x_path)

        y = None
        if cache_y_path:
            y = np.load(cache_y_path)

        ids = None
        if cache_ids_path:
            ids = np.load(cache_ids_path)

        return X, y, ids

    # If not cached, process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process volume
        vol = process_patient_volume(row)
        X_list.append(vol)

        # Store label if available
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

        # Store ID
        ids_list.append(row["BraTS21ID"])

    X = np.array(X_list, dtype=np.float32)

    y = None
    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)

    ids = None
    if len(ids_list) > 0:
        ids = np.array(ids_list)

    # Save to cache
    print(f"Saving cache to {os.path.dirname(cache_x_path)}...")
    np.save(cache_x_path, X)
    if y is not None and cache_y_path:
        np.save(cache_y_path, y)
    if ids is not None and cache_ids_path:
        np.save(cache_ids_path, ids)

    return X, y, ids


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(os.path.dirname(CACHE_TRAIN_X), exist_ok=True)

    # 1. Train Data
    X_train, y_train, _ = generate_dataset(
        TRAIN_METADATA_PATH,
        CACHE_TRAIN_X,
        CACHE_TRAIN_Y,
        load_cached_data=load_cached_data,
    )

    # 2. Val Data
    X_val, y_val, _ = generate_dataset(
        VAL_METADATA_PATH, CACHE_VAL_X, CACHE_VAL_Y, load_cached_data=load_cached_data
    )

    # 3. Test Data (No labels)
    X_test, _, ids_test = generate_dataset(
        TEST_METADATA_PATH,
        CACHE_TEST_X,
        cache_ids_path=CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)
    test_dataset = BraTSDataset(X_test, ids=ids_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data Loaded: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})"
    )

    return train_loader, val_loader, test_loader

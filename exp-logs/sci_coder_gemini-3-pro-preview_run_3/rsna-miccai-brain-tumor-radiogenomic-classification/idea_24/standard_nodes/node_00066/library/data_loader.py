import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_dicom_slice(path, img_size=224):
    """
    Reads a single DICOM file, extracts the pixel array, and resizes it.
    Returns the image and the instance number for sorting.
    """
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(float)

        # Get Instance Number for spatial sorting
        # specific tag (0020,0013)
        try:
            instance_num = int(dcm.InstanceNumber)
        except (AttributeError, ValueError):
            # Fallback if InstanceNumber is missing (unlikely in this dataset)
            # We assign a default that will be sorted by file order later if needed
            instance_num = 0

        # Resize
        img = cv2.resize(img, (img_size, img_size))

        return img, instance_num
    except Exception as e:
        # Return None if reading fails
        return None, -1


def process_modality_volume(file_paths, num_slices, img_size):
    """
    Loads, sorts, samples, and normalizes a 3D volume for a single modality.
    """
    # Explicit check for empty file lists
    if not file_paths or len(file_paths) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    slices = []

    # Load all slices
    for path in file_paths:
        full_path = os.path.join(Config.INPUT_DIR, path)
        if os.path.exists(full_path):
            img, instance_num = load_dicom_slice(full_path, img_size)
            if img is not None:
                slices.append((instance_num, img))

    # Check if we successfully loaded any slices
    if len(slices) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # Sort by Instance Number to preserve spatial coherence
    slices.sort(key=lambda x: x[0])
    volume = np.array([s[1] for s in slices])

    # High-Density Uniform Sampling from 10% to 90% depth
    total_slices = len(volume)
    start_idx = int(total_slices * 0.1)
    end_idx = int(total_slices * 0.9)

    # Ensure valid range
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_slices

    # Generate indices
    if total_slices > 0:
        indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)
        # Clamp indices just in case
        indices = np.clip(indices, 0, total_slices - 1)
        volume = volume[indices]
    else:
        # Should be caught by len(slices) == 0, but safe fallback
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # Global Volumetric Normalization (Min-Max)
    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume.astype(np.float32)


def load_patient_data(row):
    """
    Loads and stacks data for all 4 modalities for a single patient.
    Returns shape: (128, 224, 224)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    volumes = []

    for mod in modalities:
        paths = row[f"{mod}_paths"]
        # Handle potential NaN or None in dataframe
        if not isinstance(paths, list):
            paths = []

        vol = process_modality_volume(
            paths, num_slices=Config.NUM_SLICES_PER_MODALITY, img_size=Config.IMG_SIZE
        )
        volumes.append(vol)

    # Stack: [FLAIR (32), T1w (32), T1wCE (32), T2w (32)] -> (128, 224, 224)
    # Concatenate along the first dimension (depth/channel)
    full_volume = np.concatenate(volumes, axis=0)

    return full_volume


def get_dataset_arrays(metadata_path, cache_prefix, load_cached_data=True):
    """
    Handles caching logic. Loads metadata, checks for cached .npy files.
    If not found or forced reload, processes data and saves to cache.
    """
    # Define cache paths
    cache_X_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_y_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            print(f"Loading cached {cache_prefix} data from {Config.CACHE_DIR}...")
            X = np.load(cache_X_path)
            ids = np.load(cache_ids_path, allow_pickle=True)

            # y is optional (not present for test set)
            if os.path.exists(cache_y_path):
                y = np.load(cache_y_path)
            else:
                y = None

            return X, y, ids
        else:
            print(f"Cache miss for {cache_prefix}. Processing from scratch...")

    # 2. Process from scratch
    df = pd.read_parquet(metadata_path)

    # For debugging, limit size if configured
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG MODE: Limited {cache_prefix} dataset to {len(df)} samples.")

    X_list = []
    y_list = []
    ids_list = []

    print(f"Processing {len(df)} samples for {cache_prefix}...")

    for idx, row in df.iterrows():
        # Process volume
        vol = load_patient_data(row)
        X_list.append(vol)
        ids_list.append(str(row["BraTS21ID"]))

        # Process label if exists
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    print(f"Saved {cache_prefix} data to cache.")

    return X, y, ids


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is already processed and normalized in X
        img = self.X[idx]

        # Return format depends on whether labels exist
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return torch.tensor(img, dtype=torch.float32), label
        else:
            # For test set, we might need the ID to map predictions
            return torch.tensor(img, dtype=torch.float32)


def get_dataloaders(load_cached_data=True):
    """
    High-level function to generate DataLoaders for Train, Val, and Test.
    """
    # 1. Train Data
    X_train, y_train, ids_train = get_dataset_arrays(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    train_dataset = BraTSDataset(X_train, y_train, ids_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Validation Data
    X_val, y_val, ids_val = get_dataset_arrays(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    val_dataset = BraTSDataset(X_val, y_val, ids_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Test Data
    # Note: Test metadata might not have labels
    X_test, y_test, ids_test = get_dataset_arrays(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )
    test_dataset = BraTSDataset(X_test, y_test, ids_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Return ids_test separately to help with submission file creation
    return train_loader, val_loader, test_loader, ids_test

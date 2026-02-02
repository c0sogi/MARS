import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library import config

# ==========================================
# Helper Functions
# ==========================================


def numerical_sort_key(path):
    """
    Extracts the integer number from a filename like 'Image-123.dcm'
    to ensure correct numerical sorting of slices.
    """
    # Find all numbers in the path; usually the last one is the instance number
    numbers = re.findall(r"\d+", os.path.basename(path))
    if numbers:
        return int(numbers[-1])
    return 0


def load_dicom_volume(paths, img_size, num_slices):
    """
    Loads a volume from a list of DICOM paths, normalizes, samples, and resizes.

    Args:
        paths (list): List of relative file paths to DICOM files.
        img_size (int): Target spatial resolution (H, W).
        num_slices (int): Number of slices to sample.

    Returns:
        np.ndarray: A tensor of shape (num_slices, img_size, img_size) with float32 values in [0, 1].
    """
    if not paths:
        # Return empty volume if no paths (should not happen based on metadata check)
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. Sort paths numerically to ensure correct Z-ordering
    sorted_paths = sorted(paths, key=numerical_sort_key)

    # 2. Load all slices to form the 3D volume
    # We load them into a list first
    slices = []
    for p in sorted_paths:
        full_path = os.path.join(config.INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            # Convert to float for normalization
            img = dcm.pixel_array.astype(np.float32)
            slices.append(img)
        except Exception as e:
            # Skip corrupted files
            continue

    if not slices:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    volume = np.array(slices)  # Shape: (Depth, H, W)

    # 3. Global Volumetric Normalization
    # Normalize based on the global min/max of this specific modality volume
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    # 4. High-Density Uniform Sampling (10% - 90%)
    depth = volume.shape[0]
    if depth < num_slices:
        # If fewer slices than requested, we take what we have and pad or repeat
        # However, simple linspace handles indices correctly by repeating if needed
        indices = np.linspace(0, depth - 1, num_slices).astype(int)
    else:
        # Define 10% and 90% boundaries
        start = int(depth * 0.1)
        end = int(depth * 0.9)
        # Ensure end > start
        if end <= start:
            start = 0
            end = depth - 1

        indices = np.linspace(start, end, num_slices).astype(int)

    selected_slices = volume[indices]  # Shape: (num_slices, H_orig, W_orig)

    # 5. Resize each slice
    resized_volume = []
    for i in range(num_slices):
        slc = selected_slices[i]
        # cv2.resize expects (W, H)
        slc_resized = cv2.resize(
            slc, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )
        resized_volume.append(slc_resized)

    return np.array(resized_volume, dtype=np.float32)


def process_dataset(
    metadata_path, cache_x_path, cache_y_path=None, load_cached_data=True, is_test=False
):
    """
    Loads metadata, processes all patients (loading, normalizing, stacking),
    and caches the result as .npy files.
    """
    # 1. Check Cache
    if load_cached_data:
        if is_test:
            if os.path.exists(cache_x_path) and os.path.exists(config.TEST_CACHE_IDS):
                print(f"Loading cached data from {cache_x_path}...")
                X = np.load(cache_x_path)
                ids = np.load(config.TEST_CACHE_IDS, allow_pickle=True)
                return X, ids
        else:
            if os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
                print(f"Loading cached data from {cache_x_path}...")
                X = np.load(cache_x_path)
                y = np.load(cache_y_path)
                return X, y

    # 2. Load Metadata
    print(f"Processing data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    # 3. Process Patients
    X_list = []
    y_list = []
    ids_list = []

    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for idx, row in df.iterrows():
        patient_channels = []

        # Process each modality
        for mod in modalities:
            paths = row[f"{mod}_paths"]
            # Returns shape (NUM_SLICES, H, W)
            vol = load_dicom_volume(paths, config.IMG_SIZE, config.NUM_SLICES)
            patient_channels.append(vol)

        # Stack modalities along channel dimension
        # Result shape: (NUM_MODALITIES * NUM_SLICES, H, W) -> (128, 256, 256)
        patient_tensor = np.concatenate(patient_channels, axis=0)
        X_list.append(patient_tensor)

        if not is_test:
            y_list.append(row["MGMT_value"])

        ids_list.append(row["BraTS21ID"])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} patients.")

    X = np.array(X_list, dtype=np.float32)

    # 4. Save Cache
    print(f"Saving cache to {cache_x_path}...")
    np.save(cache_x_path, X)

    if is_test:
        ids = np.array(ids_list)
        np.save(config.TEST_CACHE_IDS, ids)
        return X, ids
    else:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y_path, y)
        return X, y


# ==========================================
# Dataset Class
# ==========================================


class BrainTumorDataset(Dataset):
    def __init__(self, X, y=None, ids=None, transform=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (C, H, W)
        img = self.X[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        if self.y is not None:
            label = self.y[idx]
            return img_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            # For test set, return ID as well if needed, but usually just image
            # We can return ID in a wrapper or just rely on order
            return img_tensor


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached=True
):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # --- Train ---
    print("Preparing Training Data...")
    train_X, train_y = process_dataset(
        config.TRAIN_META_PATH,
        config.TRAIN_CACHE_X,
        config.TRAIN_CACHE_Y,
        load_cached_data=load_cached,
        is_test=False,
    )
    train_dataset = BrainTumorDataset(train_X, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Validation ---
    print("Preparing Validation Data...")
    val_X, val_y = process_dataset(
        config.VAL_META_PATH,
        config.VAL_CACHE_X,
        config.VAL_CACHE_Y,
        load_cached_data=load_cached,
        is_test=False,
    )
    val_dataset = BrainTumorDataset(val_X, val_y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Test ---
    print("Preparing Test Data...")
    test_X, test_ids = process_dataset(
        config.TEST_META_PATH,
        config.TEST_CACHE_X,
        None,
        load_cached_data=load_cached,
        is_test=True,
    )
    test_dataset = BrainTumorDataset(test_X, ids=test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids

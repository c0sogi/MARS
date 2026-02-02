import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IMG_SIZE,
    NUM_SLICES,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)

# Ensure reproducible behavior
torch.manual_seed(SEED)
np.random.seed(SEED)


def extract_slice_index(filename):
    """
    Extracts the integer slice index from a DICOM filename.
    Expected format: 'Image-123.dcm' -> 123
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return -1


def load_dicom_volume(file_paths, img_size=IMG_SIZE, num_slices=NUM_SLICES):
    """
    Loads, sorts, samples, resizes, and normalizes a 3D MRI volume from a list of paths.

    Args:
        file_paths (list): List of relative file paths.
        img_size (int): Target spatial resolution (H=W).
        num_slices (int): Target depth (number of slices).

    Returns:
        np.ndarray: Preprocessed volume of shape (num_slices, img_size, img_size).
    """
    if len(file_paths) == 0:
        # Return zero volume if modality is missing
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. Sort paths by integer index
    # We construct full paths here
    full_paths = [(p, extract_slice_index(os.path.basename(p))) for p in file_paths]
    full_paths.sort(key=lambda x: x[1])
    sorted_paths = [os.path.join(INPUT_DIR, x[0]) for x in full_paths]

    total_files = len(sorted_paths)

    # 2. High-Density Uniform Sampling (10% - 90% depth)
    # We want exactly num_slices. We sample indices from the middle 80%.
    start_idx = int(total_files * 0.1)
    end_idx = int(total_files * 0.9)

    # Handle edge case where volume is too small for margins
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_files

    # Generate float indices and round to nearest integer
    indices = np.linspace(start_idx, end_idx - 1, num_slices)
    indices = np.clip(indices, 0, total_files - 1).astype(int)

    sampled_paths = [sorted_paths[i] for i in indices]

    # 3. Load, Resize, and Accumulate
    volume = []
    for p in sampled_paths:
        try:
            dcm = pydicom.dcmread(p)
            img = dcm.pixel_array.astype(np.float32)

            # Resize
            if img.shape != (img_size, img_size):
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            volume.append(img)
        except Exception:
            # Fallback for corrupt files: add zero slice
            volume.append(np.zeros((img_size, img_size), dtype=np.float32))

    volume_np = np.array(volume)  # Shape: (32, 224, 224)

    # 4. Subset-Adaptive Normalization
    # Normalize to [0, 1] based on min/max of the *sampled* volume
    v_min = volume_np.min()
    v_max = volume_np.max()

    if v_max - v_min > 0:
        volume_np = (volume_np - v_min) / (v_max - v_min)
    else:
        volume_np = np.zeros_like(volume_np)

    return volume_np


def process_patient(row):
    """
    Processes a single patient: loads 4 modalities and stacks channels.

    Returns:
        np.ndarray: Stacked volume of shape (64, 224, 224) -> (C, H, W)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Load all 4 volumes: List of 4 arrays, each (16, 224, 224)
    volumes = []
    for mod in modalities:
        paths = row.get(f"{mod}_paths", [])
        vol = load_dicom_volume(paths)
        volumes.append(vol)

    # Stack Modalities: [FLAIR, T1w, T1wCE, T2w] along channel dimension (axis 0)
    # Result shape: (16*4, 224, 224) = (64, 224, 224)
    # This groups channels by modality (e.g., all FLAIR slices, then all T1w slices)
    X = np.concatenate(volumes, axis=0)

    return X


def generate_dataset_arrays(df, desc="Data"):
    """
    Iterates over a dataframe to generate the full X, y, and ids arrays.
    """
    X_list = []
    y_list = []
    ids_list = []

    print(f"Processing {desc} ({len(df)} subjects)...")

    for idx, row in df.iterrows():
        X = process_patient(row)
        X_list.append(X)
        ids_list.append(row["BraTS21ID"])

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X_arr = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    return X_arr, y, ids


def load_data(load_cached_data=True):
    """
    Main function to load train, val, and test data.
    Handles caching to disk to save time on re-runs.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    paths = {
        "train_X": os.path.join(WORKING_DIR, "X_train.npy"),
        "train_y": os.path.join(WORKING_DIR, "y_train.npy"),
        "train_ids": os.path.join(WORKING_DIR, "ids_train.npy"),
        "val_X": os.path.join(WORKING_DIR, "X_val.npy"),
        "val_y": os.path.join(WORKING_DIR, "y_val.npy"),
        "val_ids": os.path.join(WORKING_DIR, "ids_val.npy"),
        "test_X": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_cached:
        print("Loading cached datasets from disk...")
        data = {}
        for k, v in paths.items():
            data[k] = np.load(v)
        return data

    print("Cache missing or reload requested. Generating datasets from scratch...")

    # Load Metadata
    train_df = pd.read_parquet(TRAIN_META_PATH)
    val_df = pd.read_parquet(VAL_META_PATH)
    test_df = pd.read_parquet(TEST_META_PATH)

    # Generate Arrays
    data = {}

    # Train
    t_X, t_y, t_ids = generate_dataset_arrays(train_df, "Train")
    data["train_X"] = t_X
    data["train_y"] = t_y
    data["train_ids"] = t_ids

    # Val
    v_X, v_y, v_ids = generate_dataset_arrays(val_df, "Val")
    data["val_X"] = v_X
    data["val_y"] = v_y
    data["val_ids"] = v_ids

    # Test
    te_X, _, te_ids = generate_dataset_arrays(test_df, "Test")
    data["test_X"] = te_X
    data["test_ids"] = te_ids

    # Save to cache
    print("Saving datasets to cache...")
    for k, v in data.items():
        if v is not None:
            np.save(paths[k], v)

    return data


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the 2.5D Stacked Network.
    """

    def __init__(self, X, y=None, ids=None, transform=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve pre-processed numpy arrays
        # Shape: (64, 224, 224)
        x = self.X[idx]

        # Convert to Tensor
        x = torch.from_numpy(x)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, label
        else:
            return x


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Factory function to create Train, Val, and Test DataLoaders.
    """
    data = load_data(load_cached_data=load_cached_data)

    # Train Set
    train_dataset = BraTSDataset(data["train_X"], data["train_y"], data["train_ids"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Val Set
    val_dataset = BraTSDataset(data["val_X"], data["val_y"], data["val_ids"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Set
    test_dataset = BraTSDataset(data["test_X"], None, data["test_ids"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, data["test_ids"]

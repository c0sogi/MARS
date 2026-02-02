import os
import re
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_31/"
IMG_SIZE = 256
NUM_SLICES = 16
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
METADATA_DIR = "./metadata"


def load_dicom_volume(file_paths, img_size=IMG_SIZE, num_slices=NUM_SLICES):
    """
    Loads, sorts, samples, resizes, and normalizes a DICOM volume.
    Implements External Integer Sorting, Uniform Sampling, and View-Adaptive Normalization.
    """
    # 1. Handle empty input
    if not file_paths:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 2. External Integer Sorting
    # Extract the integer ID from filenames like 'Image-10.dcm'
    def extract_id(path):
        match = re.search(r"(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else 0

    # Sort paths based on the extracted integer
    sorted_paths = sorted(file_paths, key=extract_id)

    # 3. Uniform Sampling (10%-90% range)
    total_slices = len(sorted_paths)
    if total_slices < num_slices:
        # If fewer slices than required, take all and pad later (or duplicate)
        # Here we just sample with replacement or take available indices
        indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    else:
        # Define 10% and 90% boundaries
        start_idx = int(total_slices * 0.1)
        end_idx = int(total_slices * 0.9)

        # Ensure start < end
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = total_slices - 1

        indices = np.linspace(start_idx, end_idx, num_slices, dtype=int)

    selected_paths = [sorted_paths[i] for i in indices]

    # 4. Load and Resize
    volume = []
    for p in selected_paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)

            # Resize
            if img.shape != (img_size, img_size):
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            volume.append(img)
        except Exception:
            # Fallback for corrupt files
            volume.append(np.zeros((img_size, img_size), dtype=np.float32))

    volume_array = np.array(volume, dtype=np.float32)  # Shape: (num_slices, H, W)

    # 5. View-Adaptive Per-Modality Normalization
    # Calculate min/max only on the selected slice subset
    v_min = volume_array.min()
    v_max = volume_array.max()

    if v_max - v_min > 0:
        volume_array = (volume_array - v_min) / (v_max - v_min)
    else:
        volume_array = np.zeros_like(volume_array)

    return volume_array


def get_processed_data(df, split_name, load_cached_data=True):
    """
    Orchestrates the data processing pipeline with caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    x_path = os.path.join(CACHE_DIR, f"cached_{split_name}_X.npy")
    y_path = os.path.join(CACHE_DIR, f"cached_{split_name}_y.npy")
    ids_path = os.path.join(CACHE_DIR, f"cached_{split_name}_ids.npy")

    # Check if cache exists
    if load_cached_data and os.path.exists(x_path) and os.path.exists(ids_path):
        # If y is needed (train/val) check it exists, otherwise (test) ignore
        if "MGMT_value" in df.columns and not os.path.exists(y_path):
            pass  # Cache invalid
        else:
            print(f"Loading cached {split_name} data from {CACHE_DIR}...")
            X = np.load(x_path)
            ids = np.load(ids_path, allow_pickle=True)
            y = np.load(y_path) if os.path.exists(y_path) else None
            return X, y, ids

    print(f"Processing {split_name} data from scratch...")

    X_list = []
    y_list = []
    ids_list = []

    total = len(df)

    for idx, row in df.iterrows():
        # Load each modality
        # Modality Order: FLAIR, T1w, T1wCE, T2w
        flair = load_dicom_volume(row.get("flair_paths", []))
        t1w = load_dicom_volume(row.get("t1w_paths", []))
        t1wce = load_dicom_volume(row.get("t1wce_paths", []))
        t2w = load_dicom_volume(row.get("t2w_paths", []))

        # Stack Modalities: (4 * 16, H, W) -> (64, 256, 256)
        # Modality-Grouped Stacking
        combined = np.concatenate([flair, t1w, t1wce, t2w], axis=0)

        X_list.append(combined)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # Save to cache
    np.save(x_path, X)
    np.save(ids_path, ids)
    if y is not None:
        np.save(y_path, y)

    return X, y, ids


class BraTSDataset(Dataset):
    """
    Dataset wrapper for the pre-processed tensors.
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (C, H, W)
        img = torch.tensor(self.X[idx], dtype=torch.float32)

        sample = {"image": img, "BraTS21ID": self.ids[idx]}

        if self.y is not None:
            target = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)  # (1,)
            sample["target"] = target

        return sample


def get_dataloaders(batch_size=32, load_cached_data=True):
    """
    Main function to get DataLoaders for Train, Val, and Test.
    """
    seed_everything(42)

    # Load Metadata
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # Process/Load Data
    train_X, train_y, train_ids = get_processed_data(
        train_df, "train", load_cached_data
    )
    val_X, val_y, val_ids = get_processed_data(val_df, "val", load_cached_data)
    test_X, test_y, test_ids = get_processed_data(test_df, "test", load_cached_data)

    # Create Datasets
    train_dataset = BraTSDataset(train_X, train_y, train_ids)
    val_dataset = BraTSDataset(val_X, val_y, val_ids)
    test_dataset = BraTSDataset(test_X, test_y, test_ids)

    # Create DataLoaders
    # Note: num_workers=0 is often safer for debugging, but we can use 2-4.
    # Since data is in RAM, bottlenecks are minimal.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

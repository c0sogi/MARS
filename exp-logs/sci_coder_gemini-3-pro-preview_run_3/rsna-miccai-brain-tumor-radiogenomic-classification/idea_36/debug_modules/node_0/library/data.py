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

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_36"
METADATA_DIR = "./metadata"
IMG_SIZE = 256
NUM_SLICES_TOTAL = 32
SLICES_PER_VIEW = 16
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]
SEED = 42

seed_everything(SEED)

# ==========================================
# Helper Functions
# ==========================================


def extract_slice_number(path):
    """
    Extracts the integer slice number from a DICOM filename.
    Expected format: .../Image-123.dcm
    """
    match = re.search(r"Image-(\d+)\.dcm$", path)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_file(path):
    """
    Reads a DICOM file, converts to float32, and resizes to IMG_SIZE.
    Returns a zero array if reading fails.
    """
    full_path = os.path.join(INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        return img
    except Exception as e:
        # Return zero placeholder on failure
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def get_uniform_indices(total_files, num_select):
    """
    Selects 'num_select' indices uniformly from the 10%-90% range of 'total_files'.
    """
    if total_files == 0:
        return []

    start = int(total_files * 0.1)
    end = int(total_files * 0.9)

    # Ensure valid range
    if end <= start:
        start = 0
        end = total_files

    # If we still have 0 range (e.g. 1 file), just pick that one repeatedly
    if end == start:
        indices = np.zeros(num_select, dtype=int)
    else:
        # Linspace returns evenly spaced numbers
        indices = np.linspace(start, end - 1, num_select).astype(int)

    return indices


def normalize_volume(vol):
    """
    Min-Max normalization for a volume (C, H, W) or (D, H, W).
    """
    min_val = np.min(vol)
    max_val = np.max(vol)
    if max_val - min_val > 0:
        return (vol - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(vol)


def process_patient(row, is_test=False):
    """
    Process a single patient:
    1. Load 32 slices for each modality (FLAIR, T1w, T1wCE, T2w).
    2. Split into Even/Odd streams (16 slices each).
    3. Normalize each modality within each stream independently.
    4. Stack modalities: [FLAIR, T1w, T1wCE, T2w].

    Returns:
        stream_even: (64, 256, 256)
        stream_odd:  (64, 256, 256)
        label: float (or -1 if test)
    """

    # Containers for the final stacked streams
    # Structure: List of (16, 256, 256) arrays, one per modality
    even_modality_blocks = []
    odd_modality_blocks = []

    for mod in MODALITIES:
        col_name = f"{mod}_paths"
        paths = row[col_name]

        # 1. Sort paths by integer in filename
        # Filter out None or empty paths just in case
        if paths is None:
            paths = []

        # Sort based on slice number
        paths = sorted(paths, key=extract_slice_number)

        # 2. Select 32 indices
        indices = get_uniform_indices(len(paths), NUM_SLICES_TOTAL)

        # 3. Load images
        # If no paths, we get 32 zero images
        if len(paths) == 0:
            vol_32 = np.zeros((NUM_SLICES_TOTAL, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            slices = [load_dicom_file(paths[i]) for i in indices]
            vol_32 = np.stack(slices)  # (32, 256, 256)

        # 4. Split into Even/Odd
        # Even indices: 0, 2, ..., 30
        # Odd indices: 1, 3, ..., 31
        even_indices = np.arange(0, NUM_SLICES_TOTAL, 2)
        odd_indices = np.arange(1, NUM_SLICES_TOTAL, 2)

        vol_even = vol_32[even_indices]  # (16, 256, 256)
        vol_odd = vol_32[odd_indices]  # (16, 256, 256)

        # 5. View-Adaptive Per-Modality Normalization
        vol_even_norm = normalize_volume(vol_even)
        vol_odd_norm = normalize_volume(vol_odd)

        even_modality_blocks.append(vol_even_norm)
        odd_modality_blocks.append(vol_odd_norm)

    # 6. Stack Modalities
    # Result shape: (4 * 16, 256, 256) = (64, 256, 256)
    stream_even = np.concatenate(even_modality_blocks, axis=0)
    stream_odd = np.concatenate(odd_modality_blocks, axis=0)

    # 7. Get Label
    if is_test:
        label = -1.0
    else:
        label = float(row["MGMT_value"])

    return stream_even, stream_odd, label


# ==========================================
# Caching & Loading Logic
# ==========================================


def load_dataset_split(split_name, load_cached_data=True):
    """
    Loads processed data for a split (train, val, test).
    Uses caching to avoid re-processing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    X_even_path = os.path.join(CACHE_DIR, f"X_{split_name}_even.npy")
    X_odd_path = os.path.join(CACHE_DIR, f"X_{split_name}_odd.npy")
    y_path = os.path.join(CACHE_DIR, f"y_{split_name}.npy")
    ids_path = os.path.join(CACHE_DIR, f"ids_{split_name}.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(X_even_path)
        and os.path.exists(X_odd_path)
        and os.path.exists(y_path)
        and os.path.exists(ids_path)
    ):
        print(f"Loading {split_name} data from cache...")
        X_even = np.load(X_even_path)
        X_odd = np.load(X_odd_path)
        y = np.load(y_path)
        ids = np.load(ids_path, allow_pickle=True)
        return X_even, X_odd, y, ids

    print(f"Processing {split_name} data from scratch...")

    # Load Metadata
    parquet_path = os.path.join(METADATA_DIR, f"{split_name}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    X_even_list = []
    X_odd_list = []
    y_list = []
    ids_list = []

    is_test = split_name == "test"

    for idx, row in df.iterrows():
        s_even, s_odd, label = process_patient(row, is_test=is_test)
        X_even_list.append(s_even)
        X_odd_list.append(s_odd)
        y_list.append(label)
        ids_list.append(str(row["BraTS21ID"]))

    X_even = np.array(X_even_list, dtype=np.float32)
    X_odd = np.array(X_odd_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    print(f"Saving {split_name} data to cache...")
    np.save(X_even_path, X_even)
    np.save(X_odd_path, X_odd)
    np.save(y_path, y)
    np.save(ids_path, ids)

    return X_even, X_odd, y, ids


# ==========================================
# PyTorch Dataset & Dataloaders
# ==========================================


class MGMTDataset(Dataset):
    def __init__(self, X_even, X_odd, y, ids=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return format: (even_input, odd_input, label)
        # Inputs are (64, 256, 256)
        return (
            torch.tensor(self.X_even[idx]),
            torch.tensor(self.X_odd[idx]),
            torch.tensor(self.y[idx]),
        )

    def get_ids(self):
        return self.ids


def get_dataloaders(batch_size=16, load_cached_data=True):
    """
    Returns dataloaders for train, val, and test splits.
    """
    # Load Data
    X_train_e, X_train_o, y_train, _ = load_dataset_split("train", load_cached_data)
    X_val_e, X_val_o, y_val, _ = load_dataset_split("val", load_cached_data)
    X_test_e, X_test_o, y_test, ids_test = load_dataset_split("test", load_cached_data)

    # Create Datasets
    train_dataset = MGMTDataset(X_train_e, X_train_o, y_train)
    val_dataset = MGMTDataset(X_val_e, X_val_o, y_val)
    test_dataset = MGMTDataset(X_test_e, X_test_o, y_test, ids=ids_test)

    # Create Loaders
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

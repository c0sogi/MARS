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
CACHE_DIR = "./working/idea_opt"
METADATA_DIR = "./metadata"
IMG_SIZE = 256
NUM_SLICES = 16
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
    1. Load 16 slices for each modality (FLAIR, T1w, T1wCE, T2w).
    2. Normalize each modality volume globally.
    3. Stack modalities: [FLAIR, T1w, T1wCE, T2w].

    Returns:
        X: (64, 256, 256)
        label: float (or -1 if test)
    """

    modality_blocks = []

    for mod in MODALITIES:
        col_name = f"{mod}_paths"
        paths = row[col_name]

        # 1. Sort paths by integer in filename
        if paths is None:
            paths = []
        paths = sorted(paths, key=extract_slice_number)

        # 2. Select 16 indices
        indices = get_uniform_indices(len(paths), NUM_SLICES)

        # 3. Load images
        if len(paths) == 0:
            vol = np.zeros((NUM_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            slices = [load_dicom_file(paths[i]) for i in indices]
            vol = np.stack(slices)  # (16, 256, 256)

        # 4. Global Volumetric Normalization (Cite solution_lesson_node_00015)
        # Normalize the selected subset (Cite solution_lesson_node_00073)
        vol_norm = normalize_volume(vol)
        modality_blocks.append(vol_norm)

    # 5. Stack Modalities (Cite solution_lesson_node_00018)
    # Result shape: (4 * 16, 256, 256) = (64, 256, 256)
    X = np.concatenate(modality_blocks, axis=0)

    # 6. Get Label
    if is_test:
        label = -1.0
    else:
        label = float(row["MGMT_value"])

    return X, label


# ==========================================
# Caching & Loading Logic
# ==========================================


def load_dataset_split(split_name, load_cached_data=True):
    """
    Loads processed data for a split (train, val, test).
    Uses caching to avoid re-processing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    X_path = os.path.join(CACHE_DIR, f"X_{split_name}.npy")
    y_path = os.path.join(CACHE_DIR, f"y_{split_name}.npy")
    ids_path = os.path.join(CACHE_DIR, f"ids_{split_name}.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(X_path)
        and os.path.exists(y_path)
        and os.path.exists(ids_path)
    ):
        print(f"Loading {split_name} data from cache...")
        X = np.load(X_path)
        y = np.load(y_path)
        ids = np.load(ids_path, allow_pickle=True)
        return X, y, ids

    print(f"Processing {split_name} data from scratch...")

    # Load Metadata
    parquet_path = os.path.join(METADATA_DIR, f"{split_name}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    X_list = []
    y_list = []
    ids_list = []

    is_test = split_name == "test"

    for idx, row in df.iterrows():
        X_patient, label = process_patient(row, is_test=is_test)
        X_list.append(X_patient)
        y_list.append(label)
        ids_list.append(str(row["BraTS21ID"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    print(f"Saving {split_name} data to cache...")
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(ids_path, ids)

    return X, y, ids


# ==========================================
# PyTorch Dataset & Dataloaders
# ==========================================


class MGMTDataset(Dataset):
    def __init__(self, X, y, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return format: (input, label)
        # Input is (64, 256, 256)
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
        )

    def get_ids(self):
        return self.ids


def get_dataloaders(batch_size=16, load_cached_data=True):
    """
    Returns dataloaders for train, val, and test splits.
    """
    # Load Data
    X_train, y_train, _ = load_dataset_split("train", load_cached_data)
    X_val, y_val, _ = load_dataset_split("val", load_cached_data)
    X_test, y_test, ids_test = load_dataset_split("test", load_cached_data)

    # Create Datasets
    train_dataset = MGMTDataset(X_train, y_train)
    val_dataset = MGMTDataset(X_val, y_val)
    test_dataset = MGMTDataset(X_test, y_test, ids=ids_test)

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

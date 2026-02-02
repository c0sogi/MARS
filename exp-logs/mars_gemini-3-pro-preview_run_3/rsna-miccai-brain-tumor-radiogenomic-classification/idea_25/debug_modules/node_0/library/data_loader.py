import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure OpenCV doesn't use multithreading to avoid contention with PyTorch workers
cv2.setNumThreads(0)


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the S3HD Network.
    Serves pre-processed tensors of shape (128, 224, 224).
    """

    def __init__(self, X, y, ids):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is already (C, H, W) float32
        # Convert to tensor
        img = torch.tensor(self.X[idx], dtype=torch.float32)

        # Handle target
        if self.y is not None:
            target = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            target = torch.tensor(-1.0, dtype=torch.float32)  # Placeholder for test

        return img, target, self.ids[idx]


def load_dicom_volume(file_paths):
    """
    Reads a list of DICOM files, sorts them by Instance Number,
    and returns a 3D numpy array (Depth, Height, Width).
    """
    slices = []

    if not file_paths:
        return np.array([])

    for path in file_paths:
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            dcm = pydicom.dcmread(full_path)
            # Extract pixel array and Instance Number
            # Handle cases where InstanceNumber might be missing (unlikely in this dataset but good for robustness)
            inst_num = int(dcm.InstanceNumber) if hasattr(dcm, "InstanceNumber") else -1
            img = dcm.pixel_array.astype(np.float32)
            slices.append((inst_num, img))
        except Exception:
            # Skip corrupt files
            continue

    # Sort by Instance Number to preserve spatial coherence
    slices.sort(key=lambda x: x[0])

    if not slices:
        return np.array([])

    # Stack into volume
    volume = np.stack([s[1] for s in slices])
    return volume


def process_patient(row):
    """
    Implements the S3HD preprocessing pipeline:
    1. Loads volumes for all 4 modalities.
    2. Uniformly samples 32 slices from 10%-90% depth.
    3. Resizes to 224x224.
    4. Stacks into Modality Blocks (128 channels total).
    5. Applies Global Volumetric Normalization.
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    patient_slices = []

    target_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    num_slices = Config.SLICES_PER_MODALITY  # 32

    for mod in modalities:
        col_name = f"{mod}_paths"
        paths = row[col_name] if row[col_name] is not None else []

        # Load sorted volume
        volume = load_dicom_volume(paths)

        # Prepare container for this modality
        modality_block = np.zeros(
            (num_slices, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

        if volume.shape[0] > 0:
            depth = volume.shape[0]

            # High-Density Uniform Sampling (10% to 90%)
            if depth < num_slices:
                # If fewer slices than required, take all and pad (or repeat)
                # Here we use linspace to interpolate/duplicate indices if needed
                indices = np.linspace(0, depth - 1, num_slices).astype(int)
            else:
                start = int(depth * 0.1)
                end = int(depth * 0.9)
                # Ensure end > start
                if end <= start:
                    start = 0
                    end = depth - 1
                indices = np.linspace(start, end, num_slices).astype(int)

            # Extract and resize
            for i, idx in enumerate(indices):
                slc = volume[idx]
                # cv2.resize expects (Width, Height)
                modality_block[i] = cv2.resize(slc, target_size)

        patient_slices.append(modality_block)

    # Stack along channel axis: (4, 32, 224, 224) -> (128, 224, 224)
    # This creates [Flair_0...Flair_31, T1w_0...T1w_31, ...]
    full_volume = np.concatenate(patient_slices, axis=0)

    # Global Volumetric Normalization
    min_val = np.min(full_volume)
    max_val = np.max(full_volume)

    if max_val - min_val > 0:
        full_volume = (full_volume - min_val) / (max_val - min_val)
    else:
        full_volume = np.zeros_like(full_volume)

    return full_volume


def prepare_data(df, dataset_type, load_cached_data=True):
    """
    Processes the dataframe into numpy arrays with caching.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y, ids)
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    path_X = os.path.join(Config.CACHE_DIR, f"cached_{dataset_type}_X.npy")
    path_y = os.path.join(Config.CACHE_DIR, f"cached_{dataset_type}_y.npy")
    path_ids = os.path.join(Config.CACHE_DIR, f"cached_{dataset_type}_ids.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(path_X) and os.path.exists(path_ids):
            # For test set, y might not exist or be needed, but we handle consistency
            if dataset_type == "test" or os.path.exists(path_y):
                print(f"Loading cached {dataset_type} data from {Config.CACHE_DIR}...")
                X = np.load(path_X)
                ids = np.load(path_ids)
                if dataset_type != "test":
                    y = np.load(path_y)
                else:
                    y = None
                return X, y, ids

    # 2. Process from scratch
    print(f"Processing {dataset_type} data from scratch...")

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process volume
        vol = process_patient(row)
        X_list.append(vol)

        # Store ID
        ids_list.append(row["BraTS21ID"])

        # Store Target (if exists)
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to cache
    print(f"Saving {dataset_type} data to cache...")
    np.save(path_X, X)
    np.save(path_ids, ids)
    if y is not None:
        np.save(path_y, y)

    return X, y, ids


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): If True, tries to load pre-processed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_parquet(Config.TRAIN_META_PATH)
    val_df = pd.read_parquet(Config.VAL_META_PATH)
    test_df = pd.read_parquet(Config.TEST_META_PATH)

    # Prepare Data (Process or Load Cache)
    train_X, train_y, train_ids = prepare_data(train_df, "train", load_cached_data)
    val_X, val_y, val_ids = prepare_data(val_df, "val", load_cached_data)
    test_X, test_y, test_ids = prepare_data(test_df, "test", load_cached_data)

    # Create Datasets
    train_dataset = BraTSDataset(train_X, train_y, train_ids)
    val_dataset = BraTSDataset(val_X, val_y, val_ids)
    test_dataset = BraTSDataset(test_X, test_y, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader

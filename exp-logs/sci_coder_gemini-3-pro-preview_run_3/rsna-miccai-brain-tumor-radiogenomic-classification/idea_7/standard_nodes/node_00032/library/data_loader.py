import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
from library import config

# ==========================================
# Helper Functions
# ==========================================


def load_dicom_slice(path, img_size):
    """
    Reads a single DICOM file, resizes it, and returns the pixel array and InstanceNumber.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # Resize
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Get InstanceNumber for sorting, default to -1 if missing
        instance_num = getattr(dcm, "InstanceNumber", -1)
        try:
            instance_num = int(instance_num)
        except:
            instance_num = -1

        return instance_num, img
    except Exception as e:
        # In case of corruption or read error, return None
        return None, None


def process_modality_volume(paths, input_dir, slices_per_modality, img_size):
    """
    Loads, sorts, normalizes, and samples a volume for a single modality.
    """
    if not paths or len(paths) == 0:
        return np.zeros((slices_per_modality, img_size, img_size), dtype=np.float32)

    # Read all slices
    slices = []
    # Note: paths in metadata are relative to input_dir's parent usually,
    # but config.INPUT_DIR is "./input". Metadata paths are like "train/00000/..."
    # So full path is os.path.join(config.INPUT_DIR, path)

    for rel_path in paths:
        full_path = os.path.join(config.INPUT_DIR, rel_path)
        if os.path.exists(full_path):
            idx, img = load_dicom_slice(full_path, img_size)
            if img is not None:
                slices.append((idx, img))

    if not slices:
        return np.zeros((slices_per_modality, img_size, img_size), dtype=np.float32)

    # Sort by InstanceNumber
    slices.sort(key=lambda x: x[0])
    volume = np.array([s[1] for s in slices], dtype=np.float32)

    # Global Volumetric Normalization (Min-Max)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)

    # High-Density Uniform Sampling (10% - 90%)
    depth = volume.shape[0]
    if depth > 0:
        start = int(depth * 0.1)
        end = int(depth * 0.9)

        # Handle edge cases where trimming leaves too few slices
        if end <= start:
            start = 0
            end = depth

        # Generate indices
        indices = np.linspace(start, end - 1, slices_per_modality, dtype=int)
        # Clamp indices just in case
        indices = np.clip(indices, 0, depth - 1)

        volume = volume[indices]
    else:
        volume = np.zeros((slices_per_modality, img_size, img_size), dtype=np.float32)

    return volume


def process_patient(args):
    """
    Worker function to process all modalities for a single patient.
    """
    row, slices_per_modality, img_size = args

    # Extract paths from row
    # Metadata columns: flair_paths, t1w_paths, t1wce_paths, t2w_paths
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    patient_volume = []

    for mod in modalities:
        col_name = f"{mod}_paths"
        paths = row[col_name]
        # Handle None or NaN
        if not isinstance(paths, list):
            paths = []

        mod_vol = process_modality_volume(
            paths, config.INPUT_DIR, slices_per_modality, img_size
        )
        patient_volume.append(mod_vol)

    # Stack depth-wise: (4, 32, 256, 256) -> (128, 256, 256)
    # Concatenate along the first dimension (depth/channels)
    # Each mod_vol is (32, 256, 256)
    full_volume = np.concatenate(patient_volume, axis=0)

    return full_volume


# ==========================================
# Dataset Class
# ==========================================


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to torch tensor
        # Input shape is (C, H, W) where C=128
        image = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return image, label
        else:
            # For test set, return BraTS21ID as well
            return image, self.ids[idx]


# ==========================================
# Data Loading & Caching Logic
# ==========================================


def prepare_dataset(df, split_name, load_cached_data=True):
    """
    Prepares X, y, and ids arrays. Uses caching to avoid re-processing.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    x_path = os.path.join(cache_dir, f"cached_{split_name}_X.npy")
    y_path = os.path.join(cache_dir, f"cached_{split_name}_y.npy")
    ids_path = os.path.join(cache_dir, f"cached_{split_name}_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data:
        if os.path.exists(x_path) and os.path.exists(ids_path):
            print(f"Loading cached {split_name} data from {cache_dir}...")
            X = np.load(x_path)
            ids = np.load(ids_path, allow_pickle=True)
            y = np.load(y_path) if os.path.exists(y_path) else None
            return X, y, ids

    # 2. Process from Scratch
    print(f"Processing {split_name} data from scratch...")

    # Prepare arguments for parallel processing
    # Convert dataframe rows to list of dicts/Series for iteration
    rows = [row for _, row in df.iterrows()]
    args_list = [(row, config.SLICES_PER_MODALITY, config.IMG_SIZE) for row in rows]

    # Use ProcessPoolExecutor for parallel DICOM reading/processing
    # Adjust max_workers based on CPU cores (12 vCPUs available)
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_patient, args_list))

    X = np.array(results, dtype=np.float32)
    ids = df["BraTS21ID"].values

    if "MGMT_value" in df.columns:
        y = df["MGMT_value"].values.astype(np.float32)
    else:
        y = None

    # 3. Save to Cache
    print(f"Saving {split_name} data to cache...")
    np.save(x_path, X)
    np.save(ids_path, ids)
    if y is not None:
        np.save(y_path, y)

    return X, y, ids


def get_dataloaders(load_cached_data=True, debug_sample_size=None):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_parquet(config.TRAIN_META_PATH)
    val_df = pd.read_parquet(config.VAL_META_PATH)
    test_df = pd.read_parquet(config.TEST_META_PATH)

    # Debugging: Subset data if requested
    if debug_sample_size is not None:
        train_df = train_df.iloc[:debug_sample_size]
        val_df = val_df.iloc[:debug_sample_size]
        # Keep test set full usually, or subset if really needed for pipeline check
        # But usually we want to validate pipeline on train/val

    # Prepare Data Arrays
    X_train, y_train, _ = prepare_dataset(train_df, "train", load_cached_data)
    X_val, y_val, _ = prepare_dataset(val_df, "val", load_cached_data)
    X_test, _, ids_test = prepare_dataset(test_df, "test", load_cached_data)

    # Create Datasets
    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)
    test_dataset = BraTSDataset(X_test, None, ids_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader

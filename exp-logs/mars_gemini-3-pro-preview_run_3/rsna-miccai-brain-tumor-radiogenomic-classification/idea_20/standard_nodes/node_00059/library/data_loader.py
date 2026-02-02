import os
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from library.config import Config
from library.utils import log_message


def extract_slice_number(path):
    """
    Extracts the integer slice number from a DICOM filename (e.g., 'Image-123.dcm').
    """
    try:
        # Assumes format .../Image-N.dcm
        filename = os.path.basename(path)
        name_no_ext = os.path.splitext(filename)[0]
        # Split by '-' and take the last part
        num = int(name_no_ext.split("-")[-1])
        return num
    except Exception:
        return -1


def load_dicom_slice(path, img_size):
    """
    Reads a single DICOM file, handles pixel data, and resizes it.
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        return np.zeros((img_size, img_size), dtype=np.float32)

    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception as e:
        # Return black image on failure
        return np.zeros((img_size, img_size), dtype=np.float32)


def load_patient_volume(row):
    """
    Loads and processes the full MRI volume for a single patient.

    Steps:
    1. Iterate through modalities [FLAIR, T1w, T1wCE, T2w].
    2. For each, sort files, select 32 slices uniformly from 10%-90% depth.
    3. Load and resize images.
    4. Concatenate modality blocks.
    5. Apply Global Volumetric Normalization.

    Returns:
        volume (np.ndarray): Shape (128, 256, 256), dtype float32.
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    modality_slices = []

    for mod in modalities:
        col_name = f"{mod}_paths"
        paths = row.get(col_name, [])

        # If paths is None or empty, create empty block
        if paths is None or len(paths) == 0:
            block = np.zeros(
                (Config.NUM_SLICES_PER_MODALITY, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )
            modality_slices.append(block)
            continue

        # 1. Sort paths numerically
        # We need to ensure we are sorting by the image number, not string sort
        # Paths are like 'train/00000/FLAIR/Image-1.dcm'
        valid_paths = []
        for p in paths:
            num = extract_slice_number(p)
            if num != -1:
                valid_paths.append((num, p))

        valid_paths.sort(key=lambda x: x[0])
        sorted_paths = [p for _, p in valid_paths]

        num_files = len(sorted_paths)

        if num_files == 0:
            block = np.zeros(
                (Config.NUM_SLICES_PER_MODALITY, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )
            modality_slices.append(block)
            continue

        # 2. High-Density Uniform Sampling (10% - 90%)
        # If not enough slices, take what we have and pad, or just linspace over what we have
        if num_files < Config.NUM_SLICES_PER_MODALITY:
            # If fewer files than target, duplicate/interpolate logic or just take all and pad?
            # Simple approach: linspace over available range, which will repeat some indices
            indices = np.linspace(
                0, num_files - 1, Config.NUM_SLICES_PER_MODALITY
            ).astype(int)
        else:
            # Drop top and bottom 10% to remove blank/noisy slices
            start_idx = int(num_files * 0.10)
            end_idx = int(num_files * 0.90)

            # Ensure start < end
            if start_idx >= end_idx:
                start_idx = 0
                end_idx = num_files - 1

            indices = np.linspace(
                start_idx, end_idx, Config.NUM_SLICES_PER_MODALITY
            ).astype(int)

        # 3. Load slices
        selected_slices = []
        for idx in indices:
            p = sorted_paths[idx]
            img = load_dicom_slice(p, Config.IMG_SIZE)
            selected_slices.append(img)

        # Stack slices for this modality: (32, 256, 256)
        block = np.stack(selected_slices, axis=0)
        modality_slices.append(block)

    # 4. Concatenate Modality Blocks
    # Result shape: (4 * 32, 256, 256) = (128, 256, 256)
    volume = np.concatenate(modality_slices, axis=0)

    # 5. Global Volumetric Normalization
    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)  # Avoid division by zero if volume is empty/flat

    return volume


def process_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Processes the dataframe to create X and y arrays, handling caching.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        X (np.ndarray): Input volumes.
        y (np.ndarray): Targets (or dummy if test).
        ids (np.ndarray): BraTS21IDs.
    """
    Config.setup()  # Ensure directory exists

    x_path = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_X.npy")
    y_path = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_y.npy")
    ids_path = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_ids.npy")

    # 1. Try Load
    if (
        load_cached_data
        and os.path.exists(x_path)
        and os.path.exists(y_path)
        and os.path.exists(ids_path)
    ):
        log_message(f"Loading {cache_prefix} data from cache: {Config.CACHE_DIR}")
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            ids = np.load(ids_path)
            return X, y, ids
        except Exception as e:
            log_message(f"Cache load failed ({e}). Re-processing...")

    # 2. Process
    log_message(f"Processing {cache_prefix} data from scratch ({len(df)} samples)...")

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Load Volume
        vol = load_patient_volume(row)
        X_list.append(vol)

        # Load ID
        ids_list.append(str(row["BraTS21ID"]))

        # Load Target (if exists)
        if "MGMT_value" in row:
            y_list.append(float(row["MGMT_value"]))
        else:
            y_list.append(0.5)  # Dummy for test

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    ids = np.array(ids_list)

    # 3. Save
    log_message(f"Saving {cache_prefix} data to cache...")
    np.save(x_path, X)
    np.save(y_path, y)
    np.save(ids_path, ids)

    return X, y, ids


class MGMTDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (C, H, W)
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.ids[idx],
        )


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    if os.path.exists(Config.TRAIN_META_PATH):
        train_df = pd.read_parquet(Config.TRAIN_META_PATH)
        val_df = pd.read_parquet(Config.VAL_META_PATH)
    else:
        # Fallback if metadata generation failed (should not happen based on prompt)
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()

    if os.path.exists(Config.TEST_META_PATH):
        test_df = pd.read_parquet(Config.TEST_META_PATH)
    else:
        test_df = pd.DataFrame()

    # Process Datasets
    # Train
    if not train_df.empty:
        X_train, y_train, ids_train = process_and_cache_data(
            train_df, "train", load_cached_data
        )
        train_dataset = MGMTDataset(X_train, y_train, ids_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = None

    # Val
    if not val_df.empty:
        X_val, y_val, ids_val = process_and_cache_data(val_df, "val", load_cached_data)
        val_dataset = MGMTDataset(X_val, y_val, ids_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        val_loader = None

    # Test
    if not test_df.empty:
        X_test, y_test, ids_test = process_and_cache_data(
            test_df, "test", load_cached_data
        )
        test_dataset = MGMTDataset(X_test, y_test, ids_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

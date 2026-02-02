import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma classification.
    Serves pre-processed 2.5D MRI volumes.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Input data of shape (N, 128, 256, 256).
            y (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): BraTS21IDs for test set identification.
        """
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is stored as float32 in cache
        img = self.X[idx]
        img_tensor = torch.from_numpy(img).float()

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label
        elif self.ids is not None:
            # Return ID for submission generation
            return img_tensor, str(self.ids[idx])
        else:
            return img_tensor


def load_dicom_slice(path, img_size):
    """
    Reads a DICOM file, handles errors, and resizes.
    Returns None if read fails.
    """
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array

        # Resize if necessary
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception:
        return None


def process_patient(row, img_size=256, num_slices=32):
    """
    Processes a single patient:
    1. Identifies sampled slice indices (10-90% range).
    2. Scans ALL files to find Global Min/Max for normalization.
    3. Loads sampled slices, resizes, and stacks in Interleaved format.

    Returns:
        np.ndarray: Shape (128, 256, 256)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Storage for the sampled slices: (Num_Slices, Num_Modalities, H, W)
    # This will be reshaped to (Num_Slices * Num_Modalities, H, W) -> (128, 256, 256)
    volume_buffer = np.zeros((num_slices, 4, img_size, img_size), dtype=np.float32)

    global_min = float("inf")
    global_max = float("-inf")

    # We need to temporarily store sampled images before normalization
    # Structure: sampled_data[mod_idx][slice_idx] = image_array
    sampled_data = {m_idx: {} for m_idx in range(4)}

    for m_idx, mod in enumerate(modalities):
        path_col = f"{mod}_paths"
        paths = row[path_col]

        if paths is None or len(paths) == 0:
            continue

        # 1. Determine Sampling Indices (10% - 90%)
        n_files = len(paths)
        start = int(n_files * 0.1)
        end = int(n_files * 0.9)

        # Handle edge case where 10-90% range is too small
        if end <= start:
            start = 0
            end = n_files

        # Uniformly sample indices
        if n_files > 0:
            sampled_indices = np.linspace(start, end - 1, num_slices, dtype=int)
            # Create a set for O(1) lookup during iteration
            sampled_indices_set = set(sampled_indices)
        else:
            sampled_indices = []
            sampled_indices_set = set()

        # 2. Iterate ALL files to find Global Min/Max and load Sampled Slices
        # Note: We iterate all files to strictly follow "Global Volumetric Normalization"
        for i, rel_path in enumerate(paths):
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # We must read the pixel array to find min/max
            img = load_dicom_slice(full_path, img_size)

            if img is not None:
                # Update Global Stats
                img_min = img.min()
                img_max = img.max()
                if img_min < global_min:
                    global_min = img_min
                if img_max > global_max:
                    global_max = img_max

                # If this is a sampled slice, store it
                # We need to map the current file index 'i' to the target slice index 's'
                # Since 'sampled_indices' can have duplicates (upsampling), we check all matches
                matches = np.where(sampled_indices == i)[0]
                for s_idx in matches:
                    sampled_data[m_idx][s_idx] = img

    # 3. Normalize and Fill Buffer
    # Avoid division by zero
    if global_max <= global_min:
        denom = 1.0
    else:
        denom = global_max - global_min

    for m_idx in range(4):
        for s_idx in range(num_slices):
            img = sampled_data[m_idx].get(s_idx)
            if img is not None:
                # Normalize
                norm_img = (img - global_min) / denom
                volume_buffer[s_idx, m_idx, :, :] = norm_img
            # Else: leaves as zeros (padding)

    # 4. Flatten to Interleaved Format
    # volume_buffer is (32, 4, 256, 256)
    # reshape(-1, 256, 256) -> (128, 256, 256)
    # Order becomes: S0_M0, S0_M1, S0_M2, S0_M3, S1_M0...
    final_volume = volume_buffer.reshape(-1, img_size, img_size)

    return final_volume


def prepare_data(
    meta_path, cache_x_path, cache_y_path=None, cache_ids_path=None, load_cached=True
):
    """
    Loads data from cache or processes it from scratch.
    """
    # 1. Try Loading Cache
    if load_cached:
        if os.path.exists(cache_x_path):
            print(f"Loading cached data from {cache_x_path}...")
            X = np.load(cache_x_path)

            y = None
            ids = None

            if cache_y_path and os.path.exists(cache_y_path):
                y = np.load(cache_y_path)

            if cache_ids_path and os.path.exists(cache_ids_path):
                ids = np.load(cache_ids_path)

            return X, y, ids

    # 2. Process from Scratch
    print(f"Processing data from {meta_path}...")
    df = pd.read_parquet(meta_path)

    # Debugging Subset
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling first {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process Volume
        vol = process_patient(row, Config.IMG_SIZE, Config.NUM_SLICES)
        X_list.append(vol)

        # Store Label if exists
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

        # Store ID
        if "BraTS21ID" in row:
            ids_list.append(row["BraTS21ID"])

    # Convert to Arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    ids = np.array(ids_list) if ids_list else None

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_x_path), exist_ok=True)
    np.save(cache_x_path, X)

    if y is not None and cache_y_path:
        np.save(cache_y_path, y)

    if ids is not None and cache_ids_path:
        np.save(cache_ids_path, ids)

    print(f"Data processed and saved to {Config.WORKING_DIR}")
    return X, y, ids


def get_train_loader(load_cached=True):
    X, y, _ = prepare_data(
        Config.TRAIN_META_PATH,
        Config.TRAIN_CACHE_X,
        Config.TRAIN_CACHE_Y,
        load_cached=load_cached,
    )
    dataset = BraTSDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_val_loader(load_cached=True):
    X, y, _ = prepare_data(
        Config.VAL_META_PATH,
        Config.VAL_CACHE_X,
        Config.VAL_CACHE_Y,
        load_cached=load_cached,
    )
    dataset = BraTSDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_test_loader(load_cached=True):
    X, _, ids = prepare_data(
        Config.TEST_META_PATH,
        Config.TEST_CACHE_X,
        cache_ids_path=Config.TEST_CACHE_IDS,
        load_cached=load_cached,
    )
    dataset = BraTSDataset(X, ids=ids)
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

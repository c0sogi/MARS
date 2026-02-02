import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

logger = get_logger()


def load_dicom_image(path, img_size=256):
    """
    Reads a DICOM file from the given path and resizes it.
    Returns a numpy array of shape (img_size, img_size).
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize
        img = cv2.resize(img, (img_size, img_size))
        return img
    except Exception as e:
        # In case of error (missing file, corrupt, etc.), return zeros
        # This ensures the pipeline doesn't crash for a single bad file
        return np.zeros((img_size, img_size), dtype=np.float32)


def process_patient(row):
    """
    Process a single patient's data:
    1. Identify paths for all 4 modalities.
    2. Sample 32 slices uniformly from 10%-90% depth.
    3. Load, resize, stack, normalize, and interleave.

    Returns:
        volume (np.ndarray): Shape (128, 256, 256)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # 1. Collect all paths
    # The metadata contains columns like 'flair_paths' which are lists of strings
    # We assume the lists are already sorted by file name (slice order)
    modality_paths = {}
    for mod in modalities:
        paths = row.get(f"{mod}_paths", [])
        if paths is None:
            paths = []
        modality_paths[mod] = list(paths)

    # Determine the number of slices based on the modality with the most files
    # (Usually they are registered, but we take the max to be safe or median)
    # Strategy: Use the length of the first non-empty modality to determine depth indices
    # Then apply those indices to all modalities.

    # Find a reference depth
    depths = [len(p) for p in modality_paths.values()]
    max_depth = max(depths) if depths else 0

    if max_depth == 0:
        # No data found for this patient
        return np.zeros(
            (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # 2. High-Density Uniform Sampling (10% to 90%)
    start_idx = int(max_depth * 0.1)
    end_idx = int(max_depth * 0.9)

    # Ensure strictly positive range
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = max_depth

    # Generate indices
    # We need exactly Config.NUM_SLICES (32)
    indices = np.linspace(start_idx, end_idx - 1, Config.NUM_SLICES).astype(int)

    # 3. Load Images
    # Shape: (Num_Modalities, Num_Slices, H, W) -> (4, 32, 256, 256)
    sampled_volume = np.zeros(
        (Config.NUM_MODALITIES, Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE),
        dtype=np.float32,
    )

    for m_idx, mod in enumerate(modalities):
        paths = modality_paths[mod]
        path_len = len(paths)

        for s_idx, slice_idx in enumerate(indices):
            # Handle case where a specific modality might have fewer slices than max_depth
            # We map the relative position
            if path_len > 0:
                # Map slice_idx (relative to max_depth) to this modality's depth
                # simple scaling: idx_mod = idx_max * (len_mod / len_max)
                # But usually BraTS data is co-registered. We assume direct index mapping if lengths match,
                # or clamp if they don't.

                # Robust index selection
                idx_to_load = int(slice_idx * (path_len / max_depth))
                idx_to_load = min(idx_to_load, path_len - 1)

                img = load_dicom_image(paths[idx_to_load], Config.IMG_SIZE)
                sampled_volume[m_idx, s_idx] = img
            else:
                # Modality missing, leave as zeros
                pass

    # 4. Global Volumetric Normalization
    # Normalize based on the min/max of the sampled volume
    v_min = sampled_volume.min()
    v_max = sampled_volume.max()

    if v_max - v_min > 0:
        sampled_volume = (sampled_volume - v_min) / (v_max - v_min)
    else:
        sampled_volume = np.zeros_like(sampled_volume)

    # 5. Interleaved Stacking
    # Current shape: (4, 32, 256, 256) -> (Modality, Slice, H, W)
    # Target: [Slice0_Mod0, Slice0_Mod1, Slice0_Mod2, Slice0_Mod3, Slice1_Mod0, ...]
    # Target shape: (128, 256, 256) where 128 = 32 * 4

    # Transpose to (Slice, Modality, H, W) -> (32, 4, 256, 256)
    sampled_volume = np.transpose(sampled_volume, (1, 0, 2, 3))

    # Reshape to (Slice * Modality, H, W) -> (128, 256, 256)
    final_volume = sampled_volume.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE)

    return final_volume


def get_dataset_arrays(metadata_df, cache_prefix, load_cached_data=True):
    """
    Loads dataset arrays (X, y, ids) from cache if available, otherwise processes them.

    Args:
        metadata_df (pd.DataFrame): Metadata containing paths and targets.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        X (np.ndarray): Input tensors.
        y (np.ndarray): Targets (or None for test).
        ids (np.ndarray): BraTS21IDs.
    """
    cache_X_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_y_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            logger.info(f"Loading {cache_prefix} data from cache...")
            try:
                X = np.load(
                    cache_X_path, mmap_mode="r"
                )  # Use mmap to save initial load time
                ids = np.load(cache_ids_path)

                y = None
                if "MGMT_value" in metadata_df.columns:
                    if os.path.exists(cache_y_path):
                        y = np.load(cache_y_path)

                # If shapes match expectation
                if len(X) == len(ids):
                    # Load fully into memory if RAM allows (220GB is plenty)
                    X = np.array(X)
                    return X, y, ids
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_prefix}: {e}")

    # 2. Process from scratch
    logger.info(f"Processing {cache_prefix} data from scratch...")

    ids_list = []
    X_list = []
    y_list = []

    total = len(metadata_df)
    for idx, row in metadata_df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing {cache_prefix}: {idx}/{total}")

        pid = row["BraTS21ID"]

        # Process Volume
        vol = process_patient(row)

        X_list.append(vol)
        ids_list.append(pid)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    # Convert to numpy
    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if y_list:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to cache
    logger.info(f"Saving {cache_prefix} data to cache at {Config.CACHE_DIR}...")
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids


class MGMTDataset(Dataset):
    """
    PyTorch Dataset for MGMT Promoter Methylation Prediction.
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (128, 256, 256)
        x_data = self.X[idx]

        # Convert to tensor
        x_tensor = torch.tensor(x_data, dtype=torch.float32)

        if self.y is not None:
            y_data = self.y[idx]
            y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(0)  # (1,)
            return x_tensor, y_tensor
        else:
            # For inference, return ID as well to track predictions
            pid = self.ids[idx]
            return x_tensor, pid


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    Handles metadata loading, debugging subsetting, data processing/caching.
    """
    logger.info("Initializing DataLoaders...")

    # 1. Load Metadata
    train_df = pd.read_parquet(Config.TRAIN_META_PATH)
    val_df = pd.read_parquet(Config.VAL_META_PATH)
    test_df = pd.read_parquet(Config.TEST_META_PATH)

    # 2. Handle Debug Mode
    if Config.DEBUG:
        logger.info(f"DEBUG MODE: Limiting datasets to {Config.DEBUG_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        # Test set is small, keep as is or limit
        test_df = test_df.head(Config.DEBUG_SIZE)

        # Use separate cache prefix for debug to avoid overwriting full cache
        cache_suffix = "_debug"
    else:
        cache_suffix = ""

    # 3. Get Arrays (Load or Process)
    logger.info("Preparing Training Data...")
    X_train, y_train, ids_train = get_dataset_arrays(
        train_df, f"train{cache_suffix}", load_cached_data
    )

    logger.info("Preparing Validation Data...")
    X_val, y_val, ids_val = get_dataset_arrays(
        val_df, f"val{cache_suffix}", load_cached_data
    )

    logger.info("Preparing Test Data...")
    X_test, y_test, ids_test = get_dataset_arrays(
        test_df, f"test{cache_suffix}", load_cached_data
    )

    # 4. Create Datasets
    train_dataset = MGMTDataset(X_train, y_train, ids_train)
    val_dataset = MGMTDataset(X_val, y_val, ids_val)
    test_dataset = MGMTDataset(X_test, y=None, ids=ids_test)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    logger.info(
        f"DataLoaders Ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader

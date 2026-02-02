import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import configuration and utilities
from library import config
from library import utils


def load_dicom_slice(path, img_size=256):
    """
    Reads a DICOM file, converts to float32, and resizes.
    Returns a numpy array of shape (img_size, img_size).
    """
    full_path = os.path.join(config.INPUT_DIR, path)
    try:
        ds = pydicom.dcmread(full_path)
        img = ds.pixel_array.astype(np.float32)

        # Resize if necessary
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception:
        # Fallback for corrupted files: return zero slice
        return np.zeros((img_size, img_size), dtype=np.float32)


def process_modality_volume(paths, num_slices=32, img_size=256):
    """
    Loads a volume from a list of paths, normalizes it, and samples slices.
    Returns: numpy array of shape (num_slices, img_size, img_size)
    """
    if not paths or len(paths) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. Load all available slices
    volume = []
    for p in paths:
        img = load_dicom_slice(p, img_size)
        volume.append(img)

    volume = np.array(volume)  # (D, H, W)

    # 2. Global Volumetric Normalization (per modality)
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume[:] = 0.0

    # 3. High-Density Uniform Sampling (10% to 90% depth)
    total_slices = len(volume)

    if total_slices < num_slices:
        # If fewer slices than desired, linearly interpolate indices to stretch
        indices = np.linspace(0, total_slices - 1, num_slices).astype(int)
    else:
        # Exclude top/bottom 10% noise/skull
        start = int(total_slices * 0.1)
        end = int(total_slices * 0.9)

        # Ensure start < end
        if end <= start:
            start = 0
            end = total_slices

        indices = np.linspace(start, end - 1, num_slices).astype(int)

    sampled_volume = volume[indices]  # (32, 256, 256)

    return sampled_volume


def process_patient(row, img_size=256, num_slices=32):
    """
    Processes a single patient: loads 4 modalities, samples, and interleaves.
    Returns: tensor of shape (128, 256, 256)
    """
    # Modalities
    mods = ["flair", "t1w", "t1wce", "t2w"]

    # Store processed volumes: dict of (32, 256, 256)
    volumes = {}

    for m in mods:
        # Retrieve paths from dataframe row
        paths = row[f"{m}_paths"]
        if paths is None:
            paths = []
        # Ensure paths is a list
        paths = list(paths)

        volumes[m] = process_modality_volume(
            paths, num_slices=num_slices, img_size=img_size
        )

    # Interleaved Stacking
    # Order: Slice0 [FLAIR, T1w, T1wCE, T2w], Slice1 [...], ...
    # Result channels: 32 * 4 = 128
    stacked_channels = []
    for i in range(num_slices):
        for m in mods:
            stacked_channels.append(volumes[m][i])

    # Stack into (128, 256, 256)
    X = np.stack(stacked_channels, axis=0).astype(np.float32)
    return X


def generate_dataset_arrays(df, img_size=256, num_slices=32):
    """
    Iterates over the dataframe and generates the full X, y, and ids arrays.
    """
    X_list = []
    y_list = []
    ids_list = []

    print(f"Processing {len(df)} samples...")

    for idx, row in df.iterrows():
        # Process input volume
        X = process_patient(row, img_size, num_slices)
        X_list.append(X)

        # Get ID
        ids_list.append(row["BraTS21ID"])

        # Get Target (if exists)
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])
        else:
            y_list.append(-1)  # Placeholder for test set

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    ids_arr = np.array(ids_list)

    return X_arr, y_arr, ids_arr


def get_data_for_split(split_name, meta_path, load_cached_data=True, max_samples=None):
    """
    Handles caching logic for a specific split (train, val, test).
    """
    cache_X = os.path.join(config.CACHE_DIR, f"cached_{split_name}_X.npy")
    cache_y = os.path.join(config.CACHE_DIR, f"cached_{split_name}_y.npy")
    cache_ids = os.path.join(config.CACHE_DIR, f"cached_{split_name}_ids.npy")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_X)
        and os.path.exists(cache_y)
        and os.path.exists(cache_ids)
    ):
        print(f"Loading cached {split_name} data from {config.CACHE_DIR}...")
        X = np.load(cache_X)
        y = np.load(cache_y)
        ids = np.load(cache_ids, allow_pickle=True)

        # Handle max_samples for debugging on cached data
        if max_samples is not None and len(X) > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
            ids = ids[:max_samples]

        return X, y, ids

    # 2. Process from Scratch
    print(f"Generating {split_name} data from metadata...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    # Apply max_samples before processing to save time
    if max_samples is not None:
        df = df.head(max_samples)

    X, y, ids = generate_dataset_arrays(
        df, img_size=config.IMG_SIZE, num_slices=config.NUM_SLICES_PER_MODALITY
    )

    # 3. Save to Cache
    # We only save if we processed the full dataset (max_samples is None)
    # to avoid overwriting a full cache with a partial debug cache.
    if max_samples is None and load_cached_data:
        print(f"Saving {split_name} data to cache...")
        np.save(cache_X, X)
        np.save(cache_y, y)
        np.save(cache_ids, ids)

    return X, y, ids


def get_dataloaders(
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=config.MAX_SAMPLES,
):
    """
    Main entry point. Returns train, val, test dataloaders and test IDs.
    """
    # Ensure cache dir exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Load Data Arrays
    train_X, train_y, _ = get_data_for_split(
        "train", config.TRAIN_META_PATH, load_cached_data, max_samples
    )
    val_X, val_y, _ = get_data_for_split(
        "val", config.VAL_META_PATH, load_cached_data, max_samples
    )
    test_X, _, test_ids = get_data_for_split(
        "test", config.TEST_META_PATH, load_cached_data, max_samples
    )

    # Convert to TensorDatasets
    # y needs to be (N, 1) for BCEWithLogitsLoss
    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y).unsqueeze(1)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_y).unsqueeze(1)
    )

    # Test dataset only contains X
    test_dataset = TensorDataset(torch.from_numpy(test_X))

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(
        f"Data Loaded: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})"
    )

    return train_loader, val_loader, test_loader, test_ids

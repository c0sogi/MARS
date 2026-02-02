import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from library.config import Config


def load_and_process_modality(file_paths, num_slices, img_size):
    """
    Loads DICOM files for a specific modality, sorts them by InstanceNumber,
    normalizes the volume, samples slices uniformly from 10-90% depth,
    and resizes them.
    """
    if not file_paths or len(file_paths) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. Load DICOMs and Get Instance Numbers
    slices = []
    for rel_path in file_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            dcm = pydicom.dcmread(full_path)
            # Extract InstanceNumber, default to -1 if missing
            # Some DICOMs might have InstanceNumber as a string or int
            inst_num = int(dcm.get("InstanceNumber", -1))
            pixel_array = dcm.pixel_array.astype(np.float32)
            slices.append((inst_num, pixel_array))
        except Exception:
            # Skip corrupted files or read errors
            continue

    if not slices:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 2. Sort by Instance Number
    slices.sort(key=lambda x: x[0])

    # Extract just the pixel arrays -> (D, H, W)
    volume = np.stack([s[1] for s in slices])

    # 3. Global Volumetric Normalization (per modality volume)
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    # 4. Uniform Sampling (10% - 90%)
    total_slices = volume.shape[0]

    # Define depth range
    start_idx = int(total_slices * 0.1)
    end_idx = int(total_slices * 0.9)

    # Handle edge cases where volume is small
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_slices - 1

    # Generate indices
    # We want exactly num_slices
    indices = np.linspace(start_idx, end_idx, num_slices).astype(int)

    # Clip to ensure valid bounds
    indices = np.clip(indices, 0, total_slices - 1)

    selected_slices = volume[indices]  # (num_slices, H_orig, W_orig)

    # 5. Resize
    resized_volume = []
    for i in range(num_slices):
        slc = selected_slices[i]
        # cv2.resize expects (W, H)
        slc_resized = cv2.resize(
            slc, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )
        resized_volume.append(slc_resized)

    return np.array(resized_volume, dtype=np.float32)


def process_subject(row):
    """
    Processes all modalities for a single subject and stacks them.
    Returns tensor of shape (128, 224, 224).
    """
    # Modalities: FLAIR, T1w, T1wCE, T2w
    # Order matters for "Modality-Grouped Stacking"

    flair = load_and_process_modality(
        row["flair_paths"], Config.NUM_SLICES, Config.IMG_SIZE
    )
    t1w = load_and_process_modality(
        row["t1w_paths"], Config.NUM_SLICES, Config.IMG_SIZE
    )
    t1wce = load_and_process_modality(
        row["t1wce_paths"], Config.NUM_SLICES, Config.IMG_SIZE
    )
    t2w = load_and_process_modality(
        row["t2w_paths"], Config.NUM_SLICES, Config.IMG_SIZE
    )

    # Stack: (128, 224, 224)
    # Concatenate along depth/channel dimension (axis 0)
    stacked = np.concatenate([flair, t1w, t1wce, t2w], axis=0)

    return stacked


def get_dataset_arrays(
    metadata_path, cache_prefix, load_cached_data=True, is_test=False
):
    """
    Loads metadata, processes images (or loads from cache), and returns numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_X_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_y_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try loading cache
    if (
        load_cached_data
        and os.path.exists(cache_X_path)
        and os.path.exists(cache_ids_path)
    ):
        # For test set, y might not exist
        if is_test or os.path.exists(cache_y_path):
            print(f"Loading cached {cache_prefix} data from {Config.CACHE_DIR}...")
            X = np.load(cache_X_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            if is_test:
                return X, None, ids
            else:
                y = np.load(cache_y_path)
                return X, y, ids

    # 2. Process from scratch
    print(f"Processing {cache_prefix} data from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(f"Debug mode: sampled {len(df)} rows.")

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        try:
            vol = process_subject(row)  # (128, 224, 224)
            X_list.append(vol)
            ids_list.append(row["BraTS21ID"])

            if not is_test:
                y_list.append(row["MGMT_value"])
        except Exception as e:
            print(f"Error processing subject {row['BraTS21ID']}: {e}")
            # Insert zeros to maintain alignment in case of failure
            X_list.append(
                np.zeros(
                    (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                    dtype=np.float32,
                )
            )
            ids_list.append(row["BraTS21ID"])
            if not is_test:
                y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)

    if not is_test:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y_path, y)
        return X, y, ids
    else:
        return X, None, ids


def get_dataloaders(load_cached_data=True):
    """
    Returns (train_loader, val_loader, test_loader)
    """
    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    X_train, y_train, _ = get_dataset_arrays(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)  # (N, 1)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    X_val, y_val, _ = get_dataset_arrays(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------
    X_test, _, ids_test = get_dataset_arrays(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=load_cached_data,
        is_test=True,
    )

    # Convert IDs to int64 for TensorDataset
    ids_test_int = np.array([int(x) for x in ids_test], dtype=np.int64)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(ids_test_int)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader

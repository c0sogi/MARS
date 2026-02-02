import os
import json
import numpy as np
import pandas as pd
import torch
from library.config import Config, set_seed


def get_device():
    """
    Returns the PyTorch device configured in Config.
    """
    return torch.device(Config.DEVICE)


def load_dataset(split="train", load_cached_data=True):
    """
    Loads and processes the dataset for a specific split.

    Implements caching to avoid re-processing raw JSON files.
    Constructs a 3-channel image (HH, HV, Avg) and imputes missing angles.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, tries to load from numpy cache first.

    Returns:
        tuple:
            - If split is 'train' or 'val': (X, angles, y)
            - If split is 'test': (X, angles, ids)

        Shapes:
            X: (N, 3, 75, 75) float32
            angles: (N,) float32
            y: (N,) float32 (0.0 or 1.0)
            ids: (N,) str
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_X = os.path.join(Config.CACHE_DIR, f"X_{split}.npy")
    cache_angle = os.path.join(Config.CACHE_DIR, f"angle_{split}.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"y_{split}.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"ids_{split}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if split == "test":
            if (
                os.path.exists(cache_X)
                and os.path.exists(cache_angle)
                and os.path.exists(cache_ids)
            ):
                print(f"Loading cached {split} data from {Config.CACHE_DIR}...")
                return (np.load(cache_X), np.load(cache_angle), np.load(cache_ids))
        else:
            if (
                os.path.exists(cache_X)
                and os.path.exists(cache_angle)
                and os.path.exists(cache_y)
            ):
                print(f"Loading cached {split} data from {Config.CACHE_DIR}...")
                return (np.load(cache_X), np.load(cache_angle), np.load(cache_y))

    print(f"Processing {split} data from scratch...")

    # 2. Load Metadata
    meta_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # 3. Determine Imputation Value (Median of Training Set Angles)
    # We always use the training set median to avoid data leakage
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_train_meta = pd.read_csv(train_meta_path)
        angle_median = df_train_meta["inc_angle"].median()
    else:
        # Fallback if train metadata is missing (should not happen)
        angle_median = df_meta["inc_angle"].median()

    # 4. Load Raw Data
    # Identify unique source files needed for this split
    source_files = df_meta["source_file"].unique()
    raw_data_map = {}

    for sf in source_files:
        sf_path = os.path.join(Config.INPUT_DIR, sf)
        if not os.path.exists(sf_path):
            raise FileNotFoundError(f"Raw data file not found: {sf_path}")

        with open(sf_path, "r") as f:
            # Load entire JSON list into memory
            raw_data_map[sf] = json.load(f)

    # 5. Process Samples
    n_samples = len(df_meta)

    # Initialize arrays
    X = np.zeros((n_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(n_samples, dtype=np.float32)
    ids = []
    labels = []

    for i, row in df_meta.iterrows():
        # Retrieve raw data using index from metadata
        source_file = row["source_file"]
        original_idx = int(row["original_index"])
        item = raw_data_map[source_file][original_idx]

        # Parse Bands
        # Band 1: HH
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        # Band 2: HV
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        # Band 3: Average ((HH + HV) / 2)
        b3 = (b1 + b2) / 2.0

        # Stack into (3, 75, 75)
        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        # Parse Angle
        # Metadata already has numeric conversion with NaNs
        ang = row["inc_angle"]
        if pd.isna(ang):
            angles[i] = angle_median
        else:
            angles[i] = ang

        # Parse ID
        ids.append(str(row["id"]))

        # Parse Label (if available)
        if "is_iceberg" in row:
            labels.append(row["is_iceberg"])

    # 6. Save to Cache
    np.save(cache_X, X)
    np.save(cache_angle, angles)

    if split == "test":
        ids_arr = np.array(ids)
        np.save(cache_ids, ids_arr)
        return X, angles, ids_arr
    else:
        y = np.array(labels, dtype=np.float32)
        np.save(cache_y, y)
        return X, angles, y

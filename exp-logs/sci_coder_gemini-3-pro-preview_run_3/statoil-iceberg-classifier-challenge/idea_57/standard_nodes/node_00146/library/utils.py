import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(mode, load_cached_data=True):
    """
    Loads and processes the dataset for the specified mode (train, val, test).

    This function handles:
    1. Caching: Loads from .npy files if available and requested.
    2. Data Retrieval: Reads metadata and raw JSON files.
    3. Processing: Reshapes images to (3, 75, 75) and creates the 3rd channel.
    4. Imputation: Fills missing incidence angles using the training set median.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X, angles, y, ids)
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray or None): Target labels of shape (N,) for train/val, None for test.
            ids (np.ndarray): Image IDs of shape (N,).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_X = os.path.join(Config.CACHE_DIR, f"X_{mode}.npy")
    cache_angles = os.path.join(Config.CACHE_DIR, f"angle_{mode}.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"y_{mode}.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"ids_{mode}.npy")

    # Attempt to load from cache
    if load_cached_data:
        files_exist = (
            os.path.exists(cache_X)
            and os.path.exists(cache_angles)
            and os.path.exists(cache_ids)
        )
        # For 'test', y is not expected
        if mode != "test":
            files_exist = files_exist and os.path.exists(cache_y)

        if files_exist:
            print(f"Loading {mode} data from cache...")
            X = np.load(cache_X)
            angles = np.load(cache_angles)
            ids = np.load(cache_ids)
            y = np.load(cache_y) if mode != "test" else None
            return X, angles, y, ids

    print(f"Processing {mode} data from scratch...")

    # 1. Load Metadata
    if mode == "train":
        meta_path = Config.TRAIN_CSV
    elif mode == "val":
        meta_path = Config.VAL_CSV
    elif mode == "test":
        meta_path = Config.TEST_CSV
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # 2. Load Raw JSON Data
    # Identify required source files (e.g., train.json, test.json)
    source_files = df_meta["source_file"].unique()
    raw_data_map = {}

    for source_file in source_files:
        full_path = os.path.join(Config.INPUT_DIR, source_file)
        print(f"Loading raw data from {source_file}...")
        with open(full_path, "r") as f:
            data_list = json.load(f)
            # Map ID to data item for O(1) lookup
            for item in data_list:
                raw_data_map[item["id"]] = item

    # 3. Process Data
    X_list = []
    angles_list = []
    y_list = []
    ids_list = []

    for _, row in df_meta.iterrows():
        img_id = row["id"]
        item = raw_data_map[img_id]

        # Image Processing
        # Flattened list -> 75x75 array
        band_1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        band_2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

        # Create 3rd channel: Average of Band 1 and Band 2
        avg_band = (band_1 + band_2) / 2.0

        # Stack channels: (3, 75, 75)
        img_tensor = np.stack([band_1, band_2, avg_band], axis=0)
        X_list.append(img_tensor)

        # Angle Processing
        # Use the value from metadata (which handles 'na' -> NaN conversion)
        angles_list.append(row["inc_angle"])

        # Target Processing
        if mode != "test":
            y_list.append(row["is_iceberg"])

        ids_list.append(img_id)

    # Convert to NumPy arrays
    X = np.array(X_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if mode != "test" else None

    # 4. Imputation (Median)
    # Calculate median from the TRAINING set metadata to prevent leakage
    # We load train.csv specifically for this statistic
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    train_angles = train_meta["inc_angle"].values
    median_angle = np.nanmedian(train_angles)

    # Fill NaNs in the current dataset with the training median
    nan_mask = np.isnan(angles)
    if np.any(nan_mask):
        angles[nan_mask] = median_angle

    # 5. Save to Cache
    np.save(cache_X, X)
    np.save(cache_angles, angles)
    np.save(cache_ids, ids)
    if mode != "test":
        np.save(cache_y, y)

    return X, angles, y, ids

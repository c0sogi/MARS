import os
import json
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(load_cached_data=True, debug=False):
    """
    Loads, processes, and caches the Iceberg/Ship dataset.

    Processing steps:
    1. Load metadata and raw JSON files.
    2. Construct 3-channel images (Band 1, Band 2, Mean).
    3. Impute missing incidence angles.
    4. Normalize images using Min-Max scaling derived from the training set.
    5. Cache the result as a .npz file.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.
        debug (bool): If True, subsamples the dataset for debugging purposes.

    Returns:
        dict: A dictionary containing numpy arrays for X, y, and metadata for train/val/test splits.
              Keys: 'X_train', 'y_train', 'meta_train',
                    'X_val', 'y_val', 'meta_val',
                    'X_test', 'meta_test', 'test_ids'
    """
    cache_dir = "./working/idea_14"
    os.makedirs(cache_dir, exist_ok=True)

    cache_filename = "processed_data_debug.npz" if debug else "processed_data.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}

    print("Processing data from scratch...")

    # 2. Load Metadata
    metadata_dir = "./metadata"
    train_meta = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_meta = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_meta = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    if debug:
        print("Debug mode: Subsampling data...")
        train_meta = train_meta.head(100)
        val_meta = val_meta.head(50)
        test_meta = test_meta.head(50)

    # 3. Load Raw JSON Data
    # We need to map IDs to their band data.
    input_dir = "./input"
    print("Loading raw JSON files...")
    with open(os.path.join(input_dir, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(input_dir, "test.json"), "r") as f:
        raw_test = json.load(f)

    # Create lookup dictionaries for O(1) access
    # Combine raw train and test for lookup, as metadata determines the split
    id_to_data = {item["id"]: item for item in raw_train}
    id_to_data.update({item["id"]: item for item in raw_test})

    def process_subset(meta_df, is_test=False):
        ids = meta_df["id"].values
        X_list = []
        inc_angles = []
        y_list = []

        for img_id in ids:
            item = id_to_data[img_id]

            # Extract Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)

            # Construct 3rd Channel (Mean)
            b3 = (b1 + b2) / 2.0

            # Stack to (75, 75, 3)
            img = np.stack([b1, b2, b3], axis=-1)
            X_list.append(img)

            # Extract Incidence Angle (use metadata value as it's already parsed)
            # Metadata CSV has NaN for missing values
            angle = meta_df.loc[meta_df["id"] == img_id, "inc_angle"].values[0]
            inc_angles.append(angle)

            if not is_test:
                y_list.append(item["is_iceberg"])

        X = np.array(X_list, dtype=np.float32)
        meta = np.array(inc_angles, dtype=np.float32)

        if is_test:
            return X, meta, ids
        else:
            y = np.array(y_list, dtype=np.float32)
            return X, meta, y

    print("Constructing arrays...")
    X_train, meta_train, y_train = process_subset(train_meta, is_test=False)
    X_val, meta_val, y_val = process_subset(val_meta, is_test=False)
    X_test, meta_test, test_ids = process_subset(test_meta, is_test=True)

    # 4. Impute Missing Incidence Angles
    # Calculate mean from training set
    inc_angle_mean = np.nanmean(meta_train)
    print(f"Imputing missing incidence angles with mean: {inc_angle_mean:.4f}")

    # Fill NaNs
    meta_train = np.where(np.isnan(meta_train), inc_angle_mean, meta_train)
    meta_val = np.where(np.isnan(meta_val), inc_angle_mean, meta_val)
    meta_test = np.where(np.isnan(meta_test), inc_angle_mean, meta_test)

    # 5. Normalize Images (Independent Per-Channel Min-Max Scaling)
    print("Normalizing images...")
    # Calculate stats on Train only
    # X_train shape: (N, 75, 75, 3)
    min_per_channel = np.min(X_train, axis=(0, 1, 2), keepdims=True)
    max_per_channel = np.max(X_train, axis=(0, 1, 2), keepdims=True)

    # Avoid division by zero
    denom = max_per_channel - min_per_channel
    denom[denom == 0] = 1.0

    # Apply scaling
    X_train = (X_train - min_per_channel) / denom
    X_val = (X_val - min_per_channel) / denom
    X_test = (X_test - min_per_channel) / denom

    # 6. Save to Cache
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "meta_test": meta_test,
        "test_ids": test_ids,
    }

    print(f"Saving processed data to {cache_path}...")
    np.savez(cache_path, **data_dict)

    return data_dict

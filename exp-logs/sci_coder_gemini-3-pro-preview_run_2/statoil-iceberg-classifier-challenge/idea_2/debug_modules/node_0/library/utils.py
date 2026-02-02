import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    PROCESSED_DATA_PATH,
    IMAGE_SIZE,
    MIN_DB,
    MAX_DB,
    SEED,
    WORKING_DIR,
)


def seed_everything(seed=SEED):
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


def _process_images(df):
    """
    Internal helper to process image bands from a DataFrame into a normalized 3-channel numpy array.
    Returns shape (N, 3, 75, 75).
    """
    # Stack bands into numpy arrays
    # df['band_1'] and df['band_2'] are lists of floats
    b1 = np.stack(df["band_1"].values)
    b2 = np.stack(df["band_2"].values)

    # Reshape to (N, 75, 75)
    b1 = b1.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    b2 = b2.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    # Create 3rd channel (average)
    b3 = (b1 + b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    # We use axis 1 for channels to match PyTorch convention (N, C, H, W)
    images = np.stack([b1, b2, b3], axis=1)

    # Min-Max Scaling to [0, 1]
    images = (images - MIN_DB) / (MAX_DB - MIN_DB)

    # Clip to ensure bounds
    images = np.clip(images, 0.0, 1.0)

    # Convert to float32
    return images.astype(np.float32)


def _process_angles(df, train_median=None):
    """
    Internal helper to process incidence angles.
    Replaces 'na' with NaN, converts to float, and imputes missing values.
    """
    # Convert to numeric, coercing errors to NaN
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values

    # Calculate median from current data if not provided (i.e., this is the training set)
    if train_median is None:
        train_median = np.nanmedian(angles)

    # Impute NaNs
    angles = np.where(np.isnan(angles), train_median, angles)

    return angles.astype(np.float32), train_median


def load_dataset(load_cached_data=True):
    """
    Loads the dataset.

    Logic:
    1. If load_cached_data is True and cache exists, load from disk.
    2. Otherwise, load raw JSONs and Metadata CSVs.
    3. Filter/Split data based on IDs in metadata.
    4. Process images (reshape, 3-channel, normalize).
    5. Process angles (impute missing with train median).
    6. Save processed data to disk cache.

    Returns:
        data_dict (dict): Contains:
            'X_train', 'angle_train', 'y_train', 'id_train'
            'X_val', 'angle_val', 'y_val', 'id_val'
            'X_test', 'angle_test', 'id_test'
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loading cached data from {PROCESSED_DATA_PATH}...")
        try:
            cached = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
            return {key: cached[key] for key in cached.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    meta_train = pd.read_csv(TRAIN_META_PATH)
    meta_val = pd.read_csv(VAL_META_PATH)
    meta_test = pd.read_csv(TEST_META_PATH)

    train_ids = set(meta_train["id"].values)
    val_ids = set(meta_val["id"].values)
    # Test IDs are all IDs in test.json

    # 3. Load Raw JSON Data
    # Load train.json (contains both train and val splits)
    print("Loading train.json...")
    with open(TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)
    df_raw_train = pd.DataFrame(raw_train_data)

    # Load test.json
    print("Loading test.json...")
    with open(TEST_JSON, "r") as f:
        raw_test_data = json.load(f)
    df_raw_test = pd.DataFrame(raw_test_data)

    # 4. Split Training Data into Train/Val based on Metadata IDs
    # We map IDs to the raw rows
    df_train = df_raw_train[df_raw_train["id"].isin(train_ids)].copy()
    df_val = df_raw_train[df_raw_train["id"].isin(val_ids)].copy()

    # Ensure order matches metadata is not strictly required but good practice.
    # However, raw json loading order is preserved in DataFrame.
    # We just need to ensure X, angle, y align within the split.

    # 5. Process Data
    print("Processing Training Set...")
    X_train = _process_images(df_train)
    angle_train, train_median = _process_angles(df_train, train_median=None)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    id_train = df_train["id"].values

    print(
        f"Processing Validation Set (using train median angle: {train_median:.4f})..."
    )
    X_val = _process_images(df_val)
    angle_val, _ = _process_angles(df_val, train_median=train_median)
    y_val = df_val["is_iceberg"].values.astype(np.float32)
    id_val = df_val["id"].values

    print("Processing Test Set...")
    X_test = _process_images(df_raw_test)
    angle_test, _ = _process_angles(df_raw_test, train_median=train_median)
    id_test = df_raw_test["id"].values

    # 6. Save to Cache
    print(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    np.savez(
        PROCESSED_DATA_PATH,
        X_train=X_train,
        angle_train=angle_train,
        y_train=y_train,
        id_train=id_train,
        X_val=X_val,
        angle_val=angle_val,
        y_val=y_val,
        id_val=id_val,
        X_test=X_test,
        angle_test=angle_test,
        id_test=id_test,
    )

    return {
        "X_train": X_train,
        "angle_train": angle_train,
        "y_train": y_train,
        "id_train": id_train,
        "X_val": X_val,
        "angle_val": angle_val,
        "y_val": y_val,
        "id_val": id_val,
        "X_test": X_test,
        "angle_test": angle_test,
        "id_test": id_test,
    }

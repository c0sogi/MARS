import os
import random
import json
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _process_bands(df):
    """
    Internal helper to process raw band lists into a 3-channel image tensor.
    Channels: HH (Band 1), HV (Band 2), Average ((HH+HV)/2).
    Returns: numpy array of shape (N, 3, 75, 75).
    """
    # Convert lists to numpy arrays
    # Band 1 and Band 2 are flattened 75x75 images (5625 pixels)
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Synthetic 3rd band: Average of HH and HV
    b3 = (b1 + b2) / 2.0

    # Stack along channel axis (N, C, H, W)
    X = np.stack([b1, b2, b3], axis=1)
    return X


def load_data(
    input_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_21",
    load_cached=True,
):
    """
    Loads, processes, and splits the dataset based on metadata.
    Implements caching to speed up subsequent runs.

    Returns:
        tuple: (X_train, y_train, angles_train, ids_train,
                X_val, y_val, angles_val, ids_val,
                X_test, angles_test, ids_test)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "ids_train": os.path.join(cache_dir, "ids_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "angles_val": os.path.join(cache_dir, "angles_val.npy"),
        "ids_val": os.path.join(cache_dir, "ids_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached and all_cached:
        print(f"Loading pre-processed data from cache: {cache_dir}")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v, allow_pickle=True)

        return (
            data["X_train"],
            data["y_train"],
            data["angles_train"],
            data["ids_train"],
            data["X_val"],
            data["y_val"],
            data["angles_val"],
            data["ids_val"],
            data["X_test"],
            data["angles_test"],
            data["ids_test"],
        )

    print("Processing data from scratch...")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Load Metadata
    print("Loading metadata...")
    train_meta = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_meta = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_meta = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # 2. Load Raw Data
    # train.json contains the labeled data (which we split into train/val)
    print("Loading raw train.json...")
    df_train_raw = pd.read_json(os.path.join(input_dir, "train.json"))

    print("Loading raw test.json...")
    df_test_raw = pd.read_json(os.path.join(input_dir, "test.json"))

    # 3. Process Train Split
    print("Processing train split...")
    # Use original_index from metadata to select correct rows
    train_indices = train_meta["original_index"].values
    df_train_split = df_train_raw.iloc[train_indices]

    X_train = _process_bands(df_train_split)
    y_train = train_meta["is_iceberg"].values.astype(np.float32)
    ids_train = train_meta["id"].values

    # 4. Process Val Split
    print("Processing val split...")
    val_indices = val_meta["original_index"].values
    df_val_split = df_train_raw.iloc[val_indices]

    X_val = _process_bands(df_val_split)
    y_val = val_meta["is_iceberg"].values.astype(np.float32)
    ids_val = val_meta["id"].values

    # 5. Process Test Split
    print("Processing test split...")
    test_indices = test_meta["original_index"].values
    df_test_split = df_test_raw.iloc[test_indices]

    X_test = _process_bands(df_test_split)
    ids_test = test_meta["id"].values

    # 6. Handle Incidence Angles (Imputation)
    print("Handling incidence angles...")
    # Metadata has already coerced 'na' to NaN in the 'inc_angle' column
    train_angles_raw = train_meta["inc_angle"].values
    val_angles_raw = val_meta["inc_angle"].values
    test_angles_raw = test_meta["inc_angle"].values

    # Calculate median from training set (ignoring NaNs)
    angle_median = np.nanmedian(train_angles_raw)

    # Fill NaNs with median
    angles_train = np.where(
        np.isnan(train_angles_raw), angle_median, train_angles_raw
    ).astype(np.float32)
    angles_val = np.where(
        np.isnan(val_angles_raw), angle_median, val_angles_raw
    ).astype(np.float32)
    angles_test = np.where(
        np.isnan(test_angles_raw), angle_median, test_angles_raw
    ).astype(np.float32)

    # 7. Save to Cache
    print(f"Saving processed data to cache: {cache_dir}")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["ids_train"], ids_train)

    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["angles_val"], angles_val)
    np.save(cache_files["ids_val"], ids_val)

    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return (
        X_train,
        y_train,
        angles_train,
        ids_train,
        X_val,
        y_val,
        angles_val,
        ids_val,
        X_test,
        angles_test,
        ids_test,
    )

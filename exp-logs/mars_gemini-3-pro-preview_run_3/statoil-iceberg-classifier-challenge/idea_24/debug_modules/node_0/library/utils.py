import os
import random
import numpy as np
import pandas as pd
import torch
import json


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _process_images(df_raw, indices):
    """
    Helper function to extract bands from raw dataframe based on indices,
    reshape them, and create 3-channel images (Band 1, Band 2, Average).

    Args:
        df_raw (pd.DataFrame): The raw dataframe containing 'band_1' and 'band_2'.
        indices (np.array): The original indices to extract.

    Returns:
        np.ndarray: Array of shape (N, 3, 75, 75) with float32 data.
    """
    # Select rows based on original indices
    subset = df_raw.iloc[indices]

    # Extract bands: lists of floats -> numpy array -> reshape
    # Input is flattened 5625 floats, output is 75x75
    b1 = np.array(subset["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(subset["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Create 3rd channel: average of band 1 and band 2
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 3, 75, 75) for PyTorch (C, H, W)
    images = np.stack([b1, b2, b3], axis=1)
    return images


def load_data(cache_dir="./working/idea_24", load_cached_data=True):
    """
    Loads training, validation, and test data.
    Uses metadata to split the raw JSON data.
    Implements caching to speed up subsequent runs.

    Args:
        cache_dir (str): Directory to store/load cached .npy files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, angles_train,
                X_val, y_val, angles_val,
                X_test, ids_test, angles_test)
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "angles_val": os.path.join(cache_dir, "angles_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print(f"Loading data from cache: {cache_dir}")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            angles_train = np.load(cache_files["angles_train"])
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            angles_val = np.load(cache_files["angles_val"])
            X_test = np.load(cache_files["X_test"])
            ids_test = np.load(cache_files["ids_test"], allow_pickle=True)
            angles_test = np.load(cache_files["angles_test"])
            return (
                X_train,
                y_train,
                angles_train,
                X_val,
                y_val,
                angles_val,
                X_test,
                ids_test,
                angles_test,
            )

    print("Processing data from scratch...")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load Metadata
    meta_train = pd.read_csv("./metadata/train.csv")
    meta_val = pd.read_csv("./metadata/val.csv")
    meta_test = pd.read_csv("./metadata/test.csv")

    # Load Raw Data
    # We load the full JSONs once to avoid repeated I/O
    print("Loading raw train.json...")
    df_train_raw = pd.read_json("./input/train.json")
    print("Loading raw test.json...")
    df_test_raw = pd.read_json("./input/test.json")

    # --- Process Train ---
    print("Processing Train split...")
    train_indices = meta_train["original_index"].values
    X_train = _process_images(df_train_raw, train_indices)
    y_train = meta_train["is_iceberg"].values.astype(np.float32)
    angles_train = meta_train["inc_angle"].values.astype(np.float32)

    # --- Process Val ---
    print("Processing Val split...")
    val_indices = meta_val["original_index"].values
    X_val = _process_images(df_train_raw, val_indices)
    y_val = meta_val["is_iceberg"].values.astype(np.float32)
    angles_val = meta_val["inc_angle"].values.astype(np.float32)

    # --- Process Test ---
    print("Processing Test split...")
    test_indices = meta_test["original_index"].values
    X_test = _process_images(df_test_raw, test_indices)
    ids_test = meta_test["id"].values
    angles_test = meta_test["inc_angle"].values.astype(np.float32)

    # --- Impute Angles ---
    # Calculate median from training set (ignoring NaNs)
    angle_median = np.nanmedian(angles_train)
    print(f"Imputing missing angles with median: {angle_median}")

    # Fill NaNs in all sets with the training median
    angles_train = np.nan_to_num(angles_train, nan=angle_median)
    angles_val = np.nan_to_num(angles_val, nan=angle_median)
    angles_test = np.nan_to_num(angles_test, nan=angle_median)

    # --- Save to Cache ---
    print(f"Saving processed data to {cache_dir}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["angles_val"], angles_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["angles_test"], angles_test)

    return (
        X_train,
        y_train,
        angles_train,
        X_val,
        y_val,
        angles_val,
        X_test,
        ids_test,
        angles_test,
    )

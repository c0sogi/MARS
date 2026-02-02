import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import hashlib
import json
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, seq_len, num_features).
            y (np.ndarray, optional): Target pressure of shape (num_breaths, seq_len).
            is_test (bool): Whether this is a test dataset (no targets).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def get_config_hash(config=Config):
    """
    Generates a hash based on the data configuration to ensure cache validity.
    """
    config_dict = {
        "input_features": sorted(config.INPUT_FEATURES),
        "scalable_features": sorted(config.SCALABLE_FEATURES),
        "seq_len": config.SEQ_LEN,
        "seed": config.SEED,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def engineer_features(df):
    """
    Performs physics-fidelity feature engineering.
    """
    # Ensure sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # 1. Time Delta (dt)
    # Groupby is safe but can be slow. Since data is sorted, we can use shift.
    # However, we must mask boundaries between breaths.
    # Given 80 steps per breath is constant, we can use that structure or groupby.
    # Using groupby for safety.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume (Area) Integration: cumsum(u_in * dt)
    # Vectorized calculation
    df["dv"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby("breath_id")["dv"].cumsum()

    # 3. Raw Cumulative Sum of u_in
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # 4. Lags
    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # 5. Diffs
    for diff in range(1, 3):
        df[f"u_in_diff{diff}"] = df.groupby("breath_id")["u_in"].diff(diff).fillna(0)

    # 6. Physics Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    # Avoid division by zero if C is 0 (physically impossible in this dataset, but good practice)
    # C values are 10, 20, 50.
    df["area_div_C"] = df["area"] / df["C"]

    # Drop temporary columns
    df = df.drop(columns=["dt", "dv"])

    return df


def robust_scale(
    train_data,
    val_data,
    test_data,
    scalable_features,
    feature_list,
    cache_dir,
    config_hash,
):
    """
    Applies Robust Scaling (Median/IQR) manually to avoid pickle.
    Fits on train_data, transforms all.
    """
    # Identify indices of scalable features
    scale_indices = [
        feature_list.index(f) for f in scalable_features if f in feature_list
    ]

    if not scale_indices:
        return train_data, val_data, test_data

    # Flatten train data to (N_samples, N_features) for stats calculation
    train_flat = train_data.reshape(-1, train_data.shape[-1])

    # Calculate Median and IQR
    medians = np.median(train_flat[:, scale_indices], axis=0)
    q75 = np.percentile(train_flat[:, scale_indices], 75, axis=0)
    q25 = np.percentile(train_flat[:, scale_indices], 25, axis=0)
    iqr = q75 - q25

    # Handle constant features (IQR=0) to avoid div by zero
    iqr[iqr == 0] = 1.0

    # Save scaler stats
    np.save(os.path.join(cache_dir, f"scaler_center_{config_hash}.npy"), medians)
    np.save(os.path.join(cache_dir, f"scaler_scale_{config_hash}.npy"), iqr)

    # Apply Transform
    def transform(data):
        # data shape: (N_breaths, 80, N_features)
        # We need to broadcast the stats
        # medians shape: (num_scalable,)

        # Create full-size stats arrays matching feature dimension
        full_medians = np.zeros(data.shape[-1])
        full_iqr = np.ones(data.shape[-1])

        # We only modify the scalable indices.
        # For non-scalable, we subtract 0 and divide by 1 (identity).
        # However, we must be careful not to zero out non-scalable features if we initialized full_medians with 0.
        # The logic: X_new = (X - center) / scale
        # For non-scalable: center=0, scale=1.

        full_medians[scale_indices] = medians
        full_iqr[scale_indices] = iqr

        return (data - full_medians) / full_iqr

    train_data = transform(train_data)
    val_data = transform(val_data)
    test_data = transform(test_data)

    return train_data, val_data, test_data


def prepare_datasets(config=Config, load_cached_data=True):
    """
    Orchestrates data loading, engineering, scaling, and caching.

    Returns:
        train_dataset (VentilatorDataset)
        val_dataset (VentilatorDataset)
        test_dataset (VentilatorDataset)
    """
    # Setup directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Generate Hash
    config_hash = get_config_hash(config)

    # File paths
    train_x_path = os.path.join(config.CACHE_DIR, f"train_x_{config_hash}.npy")
    train_y_path = os.path.join(config.CACHE_DIR, f"train_y_{config_hash}.npy")
    val_x_path = os.path.join(config.CACHE_DIR, f"val_x_{config_hash}.npy")
    val_y_path = os.path.join(config.CACHE_DIR, f"val_y_{config_hash}.npy")
    test_x_path = os.path.join(config.CACHE_DIR, f"test_x_{config_hash}.npy")
    test_ids_path = os.path.join(config.CACHE_DIR, f"test_ids_{config_hash}.npy")

    # Check Cache
    files_exist = all(
        os.path.exists(p)
        for p in [
            train_x_path,
            train_y_path,
            val_x_path,
            val_y_path,
            test_x_path,
            test_ids_path,
        ]
    )

    if load_cached_data and files_exist:
        print(f"Loading cached data with hash {config_hash}...")
        train_x = np.load(train_x_path)
        train_y = np.load(train_y_path)
        val_x = np.load(val_x_path)
        val_y = np.load(val_y_path)
        test_x = np.load(test_x_path)
        # We don't necessarily need test_ids for the dataset object, but good to verify
    else:
        print(f"Processing data from scratch (Hash: {config_hash})...")

        # Load Metadata CSVs
        print("Loading CSVs...")
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH)
        test_df = pd.read_csv(config.TEST_PATH)

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Extract features and targets
        features = config.INPUT_FEATURES

        # Reshaping
        # Ensure data is sorted by breath_id and time_step (engineer_features does this)
        # Check consistency
        assert (
            len(train_df) % config.SEQ_LEN == 0
        ), "Train data length not divisible by SEQ_LEN"
        assert (
            len(val_df) % config.SEQ_LEN == 0
        ), "Val data length not divisible by SEQ_LEN"
        assert (
            len(test_df) % config.SEQ_LEN == 0
        ), "Test data length not divisible by SEQ_LEN"

        print("Reshaping...")
        train_x = train_df[features].values.reshape(-1, config.SEQ_LEN, len(features))
        train_y = train_df["pressure"].values.reshape(-1, config.SEQ_LEN)

        val_x = val_df[features].values.reshape(-1, config.SEQ_LEN, len(features))
        val_y = val_df["pressure"].values.reshape(-1, config.SEQ_LEN)

        test_x = test_df[features].values.reshape(-1, config.SEQ_LEN, len(features))
        test_ids = test_df["id"].values  # Save for submission mapping

        # Scaling
        print("Scaling...")
        train_x, val_x, test_x = robust_scale(
            train_x,
            val_x,
            test_x,
            config.SCALABLE_FEATURES,
            config.INPUT_FEATURES,
            config.CACHE_DIR,
            config_hash,
        )

        # Save to Cache
        print("Saving to cache...")
        np.save(train_x_path, train_x)
        np.save(train_y_path, train_y)
        np.save(val_x_path, val_x)
        np.save(val_y_path, val_y)
        np.save(test_x_path, test_x)
        np.save(test_ids_path, test_ids)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x, is_test=True)

    print("Data preparation complete.")
    return train_dataset, val_dataset, test_dataset


def load_test_ids(config=Config):
    """
    Helper to load test IDs for submission generation.
    """
    config_hash = get_config_hash(config)
    test_ids_path = os.path.join(config.CACHE_DIR, f"test_ids_{config_hash}.npy")

    if os.path.exists(test_ids_path):
        return np.load(test_ids_path)
    else:
        # Fallback if not cached (should not happen in normal flow)
        df = pd.read_csv(config.TEST_PATH)
        return df["id"].values

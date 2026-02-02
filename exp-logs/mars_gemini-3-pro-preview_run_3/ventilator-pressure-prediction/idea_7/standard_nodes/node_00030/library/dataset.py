import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Wraps pre-processed numpy arrays for features and targets.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, 80, num_features)
            y (np.ndarray, optional): Target pressure of shape (num_breaths, 80).
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to float32 tensors
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor

        return x_tensor


def add_features(df):
    """
    Applies Feature Engineering specifically for the FCP-Net architecture.
    Calculates PID states (integral, derivative) and Physics interactions.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # 1. PID - Integral (Volume Proxy)
    # u_in_cumsum: Cumulative sum of u_in per breath
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # 2. PID - Derivative (Flow Acceleration Proxy)
    # u_in_diff1: First difference
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
    # u_in_diff2: Second difference (Jerk)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    # 3. Physics Interactions
    # R * u_in: Resistance interaction
    df["R_u_in"] = df["R"] * df["u_in"]
    # Volume / C: Compliance interaction (Elastic pressure component)
    # Note: C is in mL/cmH2O, u_in_cumsum is arbitrary units but proportional to volume
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    return df


def prepare_data(debug=False, load_cached_data=True):
    """
    Main data pipeline function.
    Handles loading, feature engineering, scaling, reshaping, and caching.

    Args:
        debug (bool): If True, subsamples data for rapid testing.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define columns to scale (exclude u_out as it is binary)
    scale_cols = [c for c in Config.FEATURE_COLS if c != "u_out"]

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE)
        and os.path.exists(Config.VAL_CACHE)
        and os.path.exists(Config.TEST_CACHE)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from parquet...")
        train_df = pd.read_parquet(Config.TRAIN_CACHE)
        val_df = pd.read_parquet(Config.VAL_CACHE)
        test_df = pd.read_parquet(Config.TEST_CACHE)
    else:
        print("Processing data from raw metadata...")
        # Load raw metadata
        train_df = pd.read_csv(Config.TRAIN_FILE)
        val_df = pd.read_csv(Config.VAL_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)

        # Debug: Subsample breaths
        if debug:
            print("Debug mode: Subsampling data...")
            train_breaths = train_df["breath_id"].unique()[:100]
            val_breaths = val_df["breath_id"].unique()[:50]
            test_breaths = test_df["breath_id"].unique()[:50]

            train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

        # Feature Engineering
        print("Applying feature engineering...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Scaling
        # Fit RobustScaler on Train only to prevent leakage
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        scaler.fit(train_df[scale_cols])

        # Transform all sets
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Save to cache
        print(f"Saving to cache: {Config.WORKING_DIR}")
        train_df.to_parquet(Config.TRAIN_CACHE)
        val_df.to_parquet(Config.VAL_CACHE)
        test_df.to_parquet(Config.TEST_CACHE)

    # Reshape to (N_breaths, 80, N_features)
    print("Reshaping data to sequences...")

    def reshape_data(df, is_test=False):
        # Sort to ensure correct order
        df = df.sort_values(["breath_id", "time_step"])

        # Extract features
        # Ensure we select columns in the exact order defined in Config
        X = df[Config.FEATURE_COLS].values
        num_breaths = len(df) // Config.SEQ_LEN

        # Reshape: (N, 80, F)
        X = X.reshape(num_breaths, Config.SEQ_LEN, -1)

        y = None
        if not is_test:
            y = df[Config.TARGET_COL].values
            y = y.reshape(num_breaths, Config.SEQ_LEN)

        return X, y

    X_train, y_train = reshape_data(train_df)
    X_val, y_val = reshape_data(val_df)
    X_test, _ = reshape_data(test_df, is_test=True)

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test, None)

    # Create DataLoaders
    # Shuffle train, but not val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

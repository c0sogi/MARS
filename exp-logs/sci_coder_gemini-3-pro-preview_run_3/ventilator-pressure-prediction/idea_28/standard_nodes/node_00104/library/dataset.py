import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.

    Provides a Dual-Stream output:
    1. Stream A: Scaled physical and kinematic features.
    2. Stream B: Raw logic/mask features (u_out) for loss gating.
    3. Target: Pressure (if available).
    """

    def __init__(self, X_stream_a, X_stream_b, y=None):
        self.X_stream_a = torch.tensor(X_stream_a, dtype=torch.float32)
        self.X_stream_b = torch.tensor(X_stream_b, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_stream_a)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_stream_a[idx], self.X_stream_b[idx], self.y[idx]
        else:
            # For test set, return dummy target or just inputs.
            # Returning 0.0 as dummy target to keep signature consistent.
            return self.X_stream_a[idx], self.X_stream_b[idx], 0.0


def add_features(df):
    """
    Generates physical and kinematic features for the ventilator dataset.
    Uses vectorized operations for efficiency.
    """
    # Ensure data is sorted by breath_id and time_step (it usually is, but safety first)
    # Note: Sorting might be expensive, assuming input is already sorted as per metadata generation.

    # 1. Time Delta (dt)
    # Global shift
    df["dt"] = df["time_step"].diff()
    # Fix boundaries: where breath_id changes, dt should be 0 (or first time step value)
    # Usually first time_step is 0, so dt=0 is appropriate.
    mask_start = df["breath_id"] != df["breath_id"].shift(1)
    df.loc[mask_start, "dt"] = 0.0
    df["dt"] = df["dt"].fillna(0.0)

    # 2. Area (Integration of u_in * dt)
    # We calculate instantaneous volume proxy
    df["vol_inst"] = df["u_in"] * df["dt"]
    # Cumulative sum grouped by breath_id
    df["area"] = df.groupby("breath_id")["vol_inst"].cumsum()

    # 3. Explicit Physics Interaction (Area / C)
    df["area_div_C"] = df["area"] / df["C"]

    # 4. Kinematics: Backward Velocity (u_in_diff)
    df["u_in_diff"] = df["u_in"].diff()
    df.loc[mask_start, "u_in_diff"] = 0.0
    df["u_in_diff"] = df["u_in_diff"].fillna(0.0)

    # 5. Kinematics: Forward Lookahead (u_in_lead 1-4)
    # We use shift(-k). We must mask where breath_id changes in the future direction.
    # breath_id shift(-k) compares current row with row k steps ahead.
    for k in range(1, 5):
        col_name = f"u_in_lead{k}"
        df[col_name] = df["u_in"].shift(-k)
        mask_end = df["breath_id"] != df["breath_id"].shift(-k)
        df.loc[mask_end, col_name] = 0.0
        df[col_name] = df[col_name].fillna(0.0)

    return df


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=4, load_cached_data=True):
    """
    Loads data, performs feature engineering, scales features, and returns DataLoaders.
    Implements caching to disk (Parquet) to speed up subsequent runs.
    """
    # Paths for cached files
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_eng.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_eng.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_eng.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached feature-engineered data...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)
    else:
        print("Processing data from raw metadata...")
        # Load raw metadata
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply Feature Engineering
        print("Generating features for Training set...")
        train_df = add_features(train_df)
        print("Generating features for Validation set...")
        val_df = add_features(val_df)
        print("Generating features for Test set...")
        test_df = add_features(test_df)

        # Save to cache
        print(f"Saving processed data to {Config.WORKING_DIR}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # --------------------------------------------------------------------------
    # Prepare Stream A (Scaled)
    # --------------------------------------------------------------------------
    print("Scaling Stream A features...")
    scaler = RobustScaler()

    # Fit scaler ONLY on training data
    X_train_a = scaler.fit_transform(train_df[Config.STREAM_A_FEATURES].values)
    X_val_a = scaler.transform(val_df[Config.STREAM_A_FEATURES].values)
    X_test_a = scaler.transform(test_df[Config.STREAM_A_FEATURES].values)

    # --------------------------------------------------------------------------
    # Prepare Stream B (Raw Logic)
    # --------------------------------------------------------------------------
    # No scaling, keep as raw values (0 or 1)
    X_train_b = train_df[Config.STREAM_B_FEATURES].values
    X_val_b = val_df[Config.STREAM_B_FEATURES].values
    X_test_b = test_df[Config.STREAM_B_FEATURES].values

    # --------------------------------------------------------------------------
    # Prepare Targets
    # --------------------------------------------------------------------------
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values
    # Test set has no target

    # --------------------------------------------------------------------------
    # Create Datasets and DataLoaders
    # --------------------------------------------------------------------------
    train_dataset = VentilatorDataset(X_train_a, X_train_b, y_train)
    val_dataset = VentilatorDataset(X_val_a, X_val_b, y_val)
    test_dataset = VentilatorDataset(X_test_a, X_test_b, y=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain batch size consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader

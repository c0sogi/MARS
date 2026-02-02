import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import set_seed


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns:
        X (torch.FloatTensor): Input features of shape (80, num_features).
        y (torch.FloatTensor): Target pressure of shape (80,).
        u_out (torch.FloatTensor): Raw u_out signal of shape (80,) for masking.
    """

    def __init__(self, X, y=None, u_out=None, is_test=False):
        self.X = X
        self.y = y
        self.u_out = u_out
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (80, features)
        x_sample = torch.tensor(self.X[idx], dtype=torch.float32)

        # u_out shape: (80,)
        # Used for loss masking (binary 0/1)
        u_out_sample = torch.tensor(self.u_out[idx], dtype=torch.float32)

        if self.is_test:
            # Dummy target for test set
            y_sample = torch.zeros(80, dtype=torch.float32)
        else:
            y_sample = torch.tensor(self.y[idx], dtype=torch.float32)

        return x_sample, y_sample, u_out_sample


def engineer_features(df):
    """
    Applies physics-based and temporal feature engineering.
    """
    # Ensure sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # --- Physics Fidelity ---
    # Calculate time delta
    df["time_delta"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Calculate Volume: Cumulative Sum of Flow (u_in) * time_delta
    # u_in is 0-100, representing valve opening. We treat it as a proxy for flow.
    df["vol_inc"] = df["u_in"] * df["time_delta"]
    df["volume"] = df.groupby("breath_id")["vol_inc"].cumsum()

    # Interaction Terms (Soft Physics)
    df["R_u_in"] = df["R"] * df["u_in"]
    # Avoid division by zero if C could be 0 (though C is 10, 20, 50 in this dataset)
    df["vol_C"] = df["volume"] / df["C"]

    # --- Temporal Dynamics ---
    # Lags for u_in
    for lag in Config.LAG_STEPS:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Differences for u_in
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    # Cleanup intermediate columns
    df = df.drop(columns=["time_delta", "vol_inc"])

    return df


def preprocess_and_reshape(train_df, val_df, test_df):
    """
    Scales features and reshapes data into (N, 80, F) tensors.
    """
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Define feature columns
    # We exclude 'id', 'breath_id', 'pressure' from X
    # We keep 'u_out' in X for Feature Completeness
    feature_cols = [
        col for col in train_df.columns if col not in ["id", "breath_id", "pressure"]
    ]

    # Identify columns to scale (exclude u_out as it is binary)
    scale_cols = [col for col in feature_cols if col != "u_out"]

    print(f"Features: {feature_cols}")
    print(f"Scaling {len(scale_cols)} features using RobustScaler...")

    scaler = RobustScaler(quantile_range=Config.ROBUST_SCALER_QUANTILE_RANGE)

    # Fit on Train, Transform All
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    # Reshape to (Num_Breaths, 80, Num_Features)
    # We assume each breath has exactly 80 steps (verified in dataset analysis)
    step_size = 80

    def reshape_data(df, is_test=False):
        # Extract X
        X = df[feature_cols].values
        num_breaths = len(df) // step_size
        X = X.reshape(num_breaths, step_size, len(feature_cols))

        # Extract u_out (raw) for masking
        u_out = df["u_out"].values.reshape(num_breaths, step_size)

        if is_test:
            y = None
        else:
            y = df["pressure"].values.reshape(num_breaths, step_size)

        return X, y, u_out

    print("Reshaping data to 3D tensors...")
    train_X, train_y, train_uout = reshape_data(train_df)
    val_X, val_y, val_uout = reshape_data(val_df)
    test_X, _, test_uout = reshape_data(test_df, is_test=True)

    return (train_X, train_y, train_uout), (val_X, val_y, val_uout), (test_X, test_uout)


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Main entry point for data loading. Handles caching and DataLoader creation.
    """
    set_seed(Config.SEED)

    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "train_uout": os.path.join(cache_dir, "train_uout.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "val_uout": os.path.join(cache_dir, "val_uout.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_uout": os.path.join(cache_dir, "test_uout.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {cache_dir}...")
        train_X = np.load(files["train_X"])
        train_y = np.load(files["train_y"])
        train_uout = np.load(files["train_uout"])
        val_X = np.load(files["val_X"])
        val_y = np.load(files["val_y"])
        val_uout = np.load(files["val_uout"])
        test_X = np.load(files["test_X"])
        test_uout = np.load(files["test_uout"])
    else:
        print("Cache missing or reload requested. Processing data from scratch...")

        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        if debug:
            print("DEBUG MODE: Truncating data...")
            # Keep only first 100 breaths (100 * 80 rows)
            train_df = train_df.iloc[: 100 * 80]
            val_df = val_df.iloc[: 100 * 80]
            test_df = test_df.iloc[: 100 * 80]

        # Process
        (
            (train_X, train_y, train_uout),
            (val_X, val_y, val_uout),
            (test_X, test_uout),
        ) = preprocess_and_reshape(train_df, val_df, test_df)

        # Save to Cache
        print(f"Saving processed data to {cache_dir}...")
        np.save(files["train_X"], train_X)
        np.save(files["train_y"], train_y)
        np.save(files["train_uout"], train_uout)
        np.save(files["val_X"], val_X)
        np.save(files["val_y"], val_y)
        np.save(files["val_uout"], val_uout)
        np.save(files["test_X"], test_X)
        np.save(files["test_uout"], test_uout)

        # Cleanup
        del train_df, val_df, test_df
        gc.collect()

    print(f"Data Shapes:")
    print(f"  Train X: {train_X.shape}, y: {train_y.shape}")
    print(f"  Val X:   {val_X.shape}, y: {val_y.shape}")
    print(f"  Test X:  {test_X.shape}")

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_y, train_uout, is_test=False)
    val_dataset = VentilatorDataset(val_X, val_y, val_uout, is_test=False)
    test_dataset = VentilatorDataset(test_X, u_out=test_uout, is_test=True)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

from library.config import Config
from library.utils import set_seed


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns:
        x (torch.Tensor): Features of shape (seq_len, num_features)
        y (torch.Tensor): Target pressure of shape (seq_len,)
        u_out (torch.Tensor): Expiratory phase flag of shape (seq_len,)
    """

    def __init__(self, X, y=None, u_out=None, is_test=False):
        self.X = X
        self.y = y
        self.u_out = u_out
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to float32 tensors
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            # For test set, return dummy targets
            # Shape of y and u_out should match seq_len (80)
            seq_len = x_tensor.shape[0]
            y_tensor = torch.zeros(seq_len, dtype=torch.float32)
            # u_out is needed for consistency, though not used for loss in test
            # We assume u_out is part of X (it is passed separately for convenience in training)
            # If u_out is provided separately:
            if self.u_out is not None:
                u_out_tensor = torch.tensor(self.u_out[idx], dtype=torch.float32)
            else:
                u_out_tensor = torch.zeros(seq_len, dtype=torch.float32)
            return x_tensor, y_tensor, u_out_tensor

        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        u_out_tensor = torch.tensor(self.u_out[idx], dtype=torch.float32)

        return x_tensor, y_tensor, u_out_tensor


def add_features(df):
    """
    Implements the Robust Physics-Fidelity Engineering pipeline.
    """
    # Ensure sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # 1. Time Delta
    # Group by breath_id to avoid diffing across breaths
    # We use a small epsilon for dt to avoid division by zero if any, though not expected here
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Time-Weighted Volume Integration
    # u_in is flow rate. Volume = integral(flow * dt)
    df["volume"] = (
        df.groupby("breath_id")
        .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
        .reset_index(level=0, drop=True)
    )

    # 3. Physics Interaction Terms
    # Resistive Pressure component ~ R * Flow
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure component ~ Volume / Compliance
    df["vol_C"] = df["volume"] / df["C"]

    # 4. Dynamics (Lags and Diffs)
    # We use groupby to ensure shifts don't leak across breaths
    # Lag features
    for lag in [1, 2]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Finite Differences (Derivatives)
    # 1st Derivative
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0)
    # 2nd Derivative
    df["u_in_diff2"] = df.groupby("breath_id")["u_in"].diff(2).fillna(0)

    # Fill any remaining NaNs (e.g. from lags/diffs at start of breath)
    df = df.fillna(0)

    return df


def prepare_data(debug=Config.DEBUG, load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, scaling, and caching.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """
    # Define cache paths
    suffix = "_debug" if debug else ""
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "train_X": os.path.join(cache_dir, f"train_X{suffix}.npy"),
        "train_y": os.path.join(cache_dir, f"train_y{suffix}.npy"),
        "train_uout": os.path.join(cache_dir, f"train_uout{suffix}.npy"),
        "val_X": os.path.join(cache_dir, f"val_X{suffix}.npy"),
        "val_y": os.path.join(cache_dir, f"val_y{suffix}.npy"),
        "val_uout": os.path.join(cache_dir, f"val_uout{suffix}.npy"),
        "test_X": os.path.join(cache_dir, f"test_X{suffix}.npy"),
        "test_uout": os.path.join(cache_dir, f"test_uout{suffix}.npy"),
        "scaler": os.path.join(cache_dir, f"scaler{suffix}.joblib"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_cached:
        print("Loading cached data from disk...")
        train_X = np.load(paths["train_X"])
        train_y = np.load(paths["train_y"])
        train_uout = np.load(paths["train_uout"])

        val_X = np.load(paths["val_X"])
        val_y = np.load(paths["val_y"])
        val_uout = np.load(paths["val_uout"])

        test_X = np.load(paths["test_X"])
        test_uout = np.load(paths["test_uout"])

    else:
        print("Processing data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_METADATA)
        val_meta = pd.read_csv(Config.VAL_METADATA)
        test_meta = pd.read_csv(Config.TEST_METADATA)

        # Load Raw Data
        train_raw = pd.read_csv(Config.TRAIN_CSV)
        test_raw = pd.read_csv(Config.TEST_CSV)

        if debug:
            # Filter for a small subset of breaths
            train_breaths = train_meta["breath_id"].unique()[:100]
            val_breaths = val_meta["breath_id"].unique()[:20]
            test_breaths = test_meta["breath_id"].unique()[:20]

            train_meta = train_meta[train_meta["breath_id"].isin(train_breaths)]
            val_meta = val_meta[val_meta["breath_id"].isin(val_breaths)]
            test_meta = test_meta[test_meta["breath_id"].isin(test_breaths)]

            train_raw = train_raw[
                train_raw["breath_id"].isin(train_breaths)
                | train_raw["breath_id"].isin(val_breaths)
            ]
            test_raw = test_raw[test_raw["breath_id"].isin(test_breaths)]

        # Split Train/Val based on metadata breath_ids
        train_breath_ids = set(train_meta["breath_id"].unique())
        val_breath_ids = set(val_meta["breath_id"].unique())

        df_train = train_raw[train_raw["breath_id"].isin(train_breath_ids)].copy()
        df_val = train_raw[train_raw["breath_id"].isin(val_breath_ids)].copy()
        df_test = test_raw.copy()

        # Feature Engineering
        print("Generating features...")
        df_train = add_features(df_train)
        df_val = add_features(df_val)
        df_test = add_features(df_test)

        # Define Feature Columns
        # We exclude ID columns and targets. We include u_out in features.
        # Continuous features for scaling
        continuous_cols = [
            "time_step",
            "u_in",
            "R",
            "C",
            "dt",
            "volume",
            "R_u_in",
            "vol_C",
            "u_in_lag1",
            "u_in_lag2",
            "u_in_diff1",
            "u_in_diff2",
        ]
        # Binary features (not scaled)
        binary_cols = ["u_out"]

        feature_cols = continuous_cols + binary_cols

        # Scaling
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        # Fit only on train
        scaler.fit(df_train[continuous_cols])

        # Transform all
        df_train[continuous_cols] = scaler.transform(df_train[continuous_cols])
        df_val[continuous_cols] = scaler.transform(df_val[continuous_cols])
        df_test[continuous_cols] = scaler.transform(df_test[continuous_cols])

        # Save Scaler
        joblib.dump(scaler, paths["scaler"])

        # Reshape to (N_breaths, Seq_Len, N_features)
        # We assume fixed sequence length of 80
        seq_len = Config.MAX_SEQ_LEN

        def reshape_data(df, is_test=False):
            # Sort to ensure order
            df = df.sort_values(["breath_id", "id"])

            # Extract arrays
            X = df[feature_cols].values
            u_out = df["u_out"].values

            num_breaths = len(df) // seq_len

            # Reshape
            X = X.reshape(num_breaths, seq_len, len(feature_cols))
            u_out = u_out.reshape(num_breaths, seq_len)

            if not is_test:
                y = df["pressure"].values
                y = y.reshape(num_breaths, seq_len)
                return X, y, u_out
            else:
                return X, None, u_out

        print("Reshaping data...")
        train_X, train_y, train_uout = reshape_data(df_train)
        val_X, val_y, val_uout = reshape_data(df_val)
        test_X, _, test_uout = reshape_data(df_test, is_test=True)

        # Cache Data
        print("Caching data...")
        np.save(paths["train_X"], train_X)
        np.save(paths["train_y"], train_y)
        np.save(paths["train_uout"], train_uout)

        np.save(paths["val_X"], val_X)
        np.save(paths["val_y"], val_y)
        np.save(paths["val_uout"], val_uout)

        np.save(paths["test_X"], test_X)
        np.save(paths["test_uout"], test_uout)

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_y, train_uout)
    val_dataset = VentilatorDataset(val_X, val_y, val_uout)
    test_dataset = VentilatorDataset(test_X, None, test_uout, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data prepared. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader

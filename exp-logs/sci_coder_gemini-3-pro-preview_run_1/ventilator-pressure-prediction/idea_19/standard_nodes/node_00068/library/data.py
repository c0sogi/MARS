import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Holds pre-processed tensors for features, targets, and control masks.
    """

    def __init__(self, X, u_out, y=None, ids=None):
        self.X = X
        self.u_out = u_out
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "u_out": torch.tensor(self.u_out[idx], dtype=torch.float32),
        }
        if self.y is not None:
            data["y"] = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.ids is not None:
            data["ids"] = torch.tensor(self.ids[idx], dtype=torch.long)
        return data


def add_features(df):
    """
    Adds physics-based and dynamic features to the dataframe.
    """
    # Ensure sorted order for time-series calculations
    # (Metadata generation usually ensures this, but safety first)
    # df = df.sort_values(['breath_id', 'time_step'])

    # --- Time Delta ---
    # Calculate dt: time_step[t] - time_step[t-1]
    # We use groupby to handle breath boundaries correctly (first dt is 0 or NaN)
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # --- Physics: Volume ---
    # Volume is integral of flow (u_in) over time
    # volume = cumsum(u_in * dt)
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # --- Physics: Interactions ---
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["volume"] / df["C"]

    # --- Dynamics: Lags ---
    for lag in Config.LAGS:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # --- Dynamics: Differences ---
    # 1st Difference
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0)
    # 2nd Difference (Acceleration) -> Diff of Diff
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff(1).fillna(0)

    return df


def get_dataloaders(
    load_cached_data=True,
    debug=Config.DEBUG,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to load, process, and return DataLoaders.
    Handles caching, scaling, and reshaping.
    """
    seed_everything()

    # Define feature columns (Continuous features to be scaled)
    # u_out is strictly excluded from this list and handled separately
    feature_cols = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_diff1",
        "u_in_diff2",
    ]
    # Add lag columns dynamically
    feature_cols += [f"u_in_lag{lag}" for lag in Config.LAGS]

    # Generate Cache Hash
    # Hash depends on seed, debug status, scaler params, and feature list
    cache_config_str = (
        f"{Config.SEED}_{debug}_{Config.ROBUST_SCALER_QUANTILE_RANGE}_{feature_cols}"
    )
    cache_hash = hashlib.md5(cache_config_str.encode()).hexdigest()

    # Define Cache Paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, f"train_X_{cache_hash}.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, f"train_y_{cache_hash}.npy"),
        "train_uout": os.path.join(Config.CACHE_DIR, f"train_uout_{cache_hash}.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, f"val_X_{cache_hash}.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, f"val_y_{cache_hash}.npy"),
        "val_uout": os.path.join(Config.CACHE_DIR, f"val_uout_{cache_hash}.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, f"test_X_{cache_hash}.npy"),
        "test_uout": os.path.join(Config.CACHE_DIR, f"test_uout_{cache_hash}.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, f"test_ids_{cache_hash}.npy"),
    }

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.CACHE_DIR} (Hash: {cache_hash})...")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        train_uout = np.load(cache_files["train_uout"])

        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        val_uout = np.load(cache_files["val_uout"])

        test_X = np.load(cache_files["test_X"])
        test_uout = np.load(cache_files["test_uout"])
        test_ids = np.load(cache_files["test_ids"])

    else:
        print("Processing data from scratch...")

        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug Mode: Slice data
        if debug:
            print(f"DEBUG MODE: Using only {Config.DEBUG_BREATHS} breaths per split.")
            train_bids = train_df["breath_id"].unique()[: Config.DEBUG_BREATHS]
            val_bids = val_df["breath_id"].unique()[: Config.DEBUG_BREATHS]
            test_bids = test_df["breath_id"].unique()[: Config.DEBUG_BREATHS]

            train_df = train_df[train_df["breath_id"].isin(train_bids)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_bids)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_bids)].copy()

        # Feature Engineering
        print("Applying Feature Engineering...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Scaling
        # Fit scaler ONLY on training data features
        print("Fitting RobustScaler...")
        scaler = RobustScaler(quantile_range=Config.ROBUST_SCALER_QUANTILE_RANGE)
        scaler.fit(train_df[feature_cols])

        print("Transforming data...")
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        # Reshaping Helper
        def reshape_to_sequence(df, is_test=False):
            # Assumes 80 time steps per breath
            # Calculate number of breaths
            n_breaths = len(df) // 80

            # Reshape Features: (N, 80, F)
            X = df[feature_cols].values.reshape(n_breaths, 80, len(feature_cols))

            # Reshape u_out: (N, 80)
            u_out = df["u_out"].values.reshape(n_breaths, 80)

            if is_test:
                y = None
                ids = df["id"].values.reshape(n_breaths, 80)
            else:
                y = df["pressure"].values.reshape(n_breaths, 80)
                ids = None

            return X, u_out, y, ids

        print("Reshaping tensors...")
        train_X, train_uout, train_y, _ = reshape_to_sequence(train_df)
        val_X, val_uout, val_y, _ = reshape_to_sequence(val_df)
        test_X, test_uout, _, test_ids = reshape_to_sequence(test_df, is_test=True)

        # Save to Cache
        print("Saving processed data to cache...")
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["train_uout"], train_uout)

        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["val_uout"], val_uout)

        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_uout"], test_uout)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_uout, train_y)
    val_dataset = VentilatorDataset(val_X, val_uout, val_y)
    test_dataset = VentilatorDataset(test_X, test_uout, ids=test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data Loaders Ready. Train Shape: {train_X.shape}")
    return train_loader, val_loader, test_loader

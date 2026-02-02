import os
import hashlib
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves sequences of breath data.
    """

    def __init__(self, X, u_out, y=None):
        self.X = X
        self.u_out = u_out
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Inputs are converted to float32 tensors
        x_val = torch.tensor(self.X[idx], dtype=torch.float32)
        u_out_val = torch.tensor(self.u_out[idx], dtype=torch.float32)

        if self.y is not None:
            y_val = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_val, u_out_val, y_val

        return x_val, u_out_val


def compute_features(df):
    """
    Computes physics-based and dynamic features for the ventilator dataset.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # --- Dynamic Features (Lags & Diffs) ---
    # Groupby shift is used to respect breath boundaries
    for i in range(1, 5):
        df[f"u_in_lag{i}"] = df.groupby("breath_id")["u_in"].shift(i).fillna(0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - df.groupby("breath_id")["u_in_diff1"].shift(
        1
    ).fillna(0)

    # --- Physics Integration (Volume) ---
    # Calculate dt (time delta)
    # We shift time_step by 1 within each breath to get the previous time
    df["time_prev"] = (
        df.groupby("breath_id")["time_step"].shift(1).fillna(df["time_step"])
    )
    df["dt"] = df["time_step"] - df["time_prev"]

    # Volume = Cumulative Sum of (Flow * dt)
    # u_in is proportional to flow
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # --- Interaction Terms ---
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["volume"] / df["C"]

    # Cleanup temporary columns
    df = df.drop(["time_prev", "dt"], axis=1)

    return df


def get_data_loaders(load_cached_data=True, debug=None):
    """
    Orchestrates data loading, feature engineering, scaling, caching, and DataLoader creation.
    """
    if debug is None:
        debug = Config.DEBUG

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define features to be used and scaled
    # Base features + Engineered features
    cont_features = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_out",  # Cite solution_lesson_node_00086: Include u_out in input features
    ]

    # Generate a hash for caching based on configuration and feature set version
    # This ensures that if we change debug mode or feature logic (version string), we recompute.
    feature_version = "v2_uout_included"
    cache_hash = hashlib.md5(
        f"{feature_version}_{debug}_{Config.EXPERIMENT_NAME}".encode()
    ).hexdigest()

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, f"train_X_{cache_hash}.npy"),
        "train_uout": os.path.join(Config.CACHE_DIR, f"train_uout_{cache_hash}.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, f"train_y_{cache_hash}.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, f"val_X_{cache_hash}.npy"),
        "val_uout": os.path.join(Config.CACHE_DIR, f"val_uout_{cache_hash}.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, f"val_y_{cache_hash}.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, f"test_X_{cache_hash}.npy"),
        "test_uout": os.path.join(Config.CACHE_DIR, f"test_uout_{cache_hash}.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, f"test_ids_{cache_hash}.npy"),
    }

    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        train_X = np.load(cache_files["train_X"])
        train_uout = np.load(cache_files["train_uout"])
        train_y = np.load(cache_files["train_y"])

        val_X = np.load(cache_files["val_X"])
        val_uout = np.load(cache_files["val_uout"])
        val_y = np.load(cache_files["val_y"])

        test_X = np.load(cache_files["test_X"])
        test_uout = np.load(cache_files["test_uout"])
        # test_ids are saved for reference/submission generation but not needed for the loader

    else:
        print("Cache not found or invalid. Processing data from scratch...")

        # Load raw data using metadata paths
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug mode: sample a subset of breaths
        if debug:
            print("DEBUG MODE: Truncating datasets...")
            steps_per_breath = 80
            train_df = train_df.iloc[: 100 * steps_per_breath]
            val_df = val_df.iloc[: 50 * steps_per_breath]
            test_df = test_df.iloc[: 50 * steps_per_breath]

        # Feature Engineering
        print("Engineering features (Physics + Dynamics)...")
        train_df = compute_features(train_df)
        val_df = compute_features(val_df)
        test_df = compute_features(test_df)

        # Cite solution_lesson_node_00058: Segregate control variables for logic
        # Preserve raw u_out for masking before it gets scaled in cont_features
        train_df["u_out_raw"] = train_df["u_out"]
        val_df["u_out_raw"] = val_df["u_out"]
        test_df["u_out_raw"] = test_df["u_out"]

        # Scaling
        # We fit RobustScaler only on training data to prevent leakage
        print("Scaling continuous features...")
        scaler = RobustScaler()
        train_df[cont_features] = scaler.fit_transform(train_df[cont_features])
        val_df[cont_features] = scaler.transform(val_df[cont_features])
        test_df[cont_features] = scaler.transform(test_df[cont_features])

        # Reshaping to (N_breaths, 80, N_features)
        print("Reshaping data for LSTM input...")
        steps_per_breath = 80

        def reshape_dataset(df, is_test=False):
            # Ensure strict sorting
            df = df.sort_values(["breath_id", "time_step"])
            n_breaths = len(df) // steps_per_breath

            # Extract arrays
            X = df[cont_features].values.reshape(
                n_breaths, steps_per_breath, len(cont_features)
            )
            # Use raw u_out for the masking tensor
            u_out = df["u_out_raw"].values.reshape(n_breaths, steps_per_breath)

            if not is_test:
                y = df["pressure"].values.reshape(n_breaths, steps_per_breath)
                return X, u_out, y, None
            else:
                ids = df["id"].values.reshape(n_breaths, steps_per_breath)
                return X, u_out, None, ids

        train_X, train_uout, train_y, _ = reshape_dataset(train_df)
        val_X, val_uout, val_y, _ = reshape_dataset(val_df)
        test_X, test_uout, _, test_ids = reshape_dataset(test_df, is_test=True)

        # Save to cache
        print("Saving processed data to cache...")
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_uout"], train_uout)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_uout"], val_uout)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_uout"], test_uout)
        np.save(cache_files["test_ids"], test_ids)

        # Memory cleanup
        del train_df, val_df, test_df
        gc.collect()

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_uout, train_y)
    val_dataset = VentilatorDataset(val_X, val_uout, val_y)
    test_dataset = VentilatorDataset(test_X, test_uout)

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
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
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, test_loader

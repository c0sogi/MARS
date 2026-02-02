import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns (features, target, u_out) for training/validation.
    Returns (features, id, u_out) for testing (if targets are missing).
    """

    def __init__(self, X, u_out, y=None, ids=None, is_test=False):
        self.X = X
        self.u_out = u_out
        self.y = y
        self.ids = ids
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Features: (80, n_features)
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        # u_out: (80,) - Used for masking loss
        u = torch.tensor(self.u_out[idx], dtype=torch.float32)

        if self.is_test:
            # For test, we might need the breath_id or similar for submission reconstruction,
            # but usually we predict in order. Here we return x and u.
            # If ids are provided (e.g. for debugging), we can return them,
            # but standard loop expects tensors.
            return x, u
        else:
            # Target: (80,)
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y, u


def add_features(df):
    """
    Implements the feature engineering pipeline described in Idea 25.

    Kinematics:
    - Backward Velocity: u_in(t) - u_in(t-1)
    - Forward Lookahead: u_in(t+1)...u_in(t+4)

    Physics:
    - dt
    - Area (Integral of u_in * dt)
    - Interactions: R*u_in, Area/C

    Exclusions:
    - raw time_step
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"])

    # 1. Time Delta (dt)
    # Group by breath_id to ensure diff doesn't cross breaths
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Physics: Volume Approximation (Area)
    # Area = Cumulative Sum of (u_in * dt)
    # We use a vectorized approach for speed
    df["vol_inst"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby("breath_id")["vol_inst"].cumsum()

    # 3. Kinematics: Backward Velocity (Momentum) & Acceleration
    # u_in_lag1
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    # u_in_diff2 (Acceleration) - Cite Lesson 00052, 00090
    df["u_in_diff2"] = df["u_in_diff1"] - df.groupby("breath_id")["u_in_diff1"].shift(
        1
    ).fillna(0)

    # 4. Kinematics: Forward Lookahead (Intent)
    # u_in(t+1) ... u_in(t+4)
    for i in range(1, Config.LEAD_STEPS + 1):
        df[f"u_in_lead{i}"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0)

    # 5. Interactions
    df["R_uin"] = df["R"] * df["u_in"]
    df["area_C"] = df["area"] / df["C"]

    # 6. Feature Selection
    # We keep u_out as a feature as well
    features = [
        "u_in",
        "u_out",
        "R",
        "C",  # Raw Controls/Attributes
        "dt",  # Time physics
        "u_in_diff1",  # Momentum
        "u_in_diff2",  # Acceleration
        "area",
        "R_uin",
        "area_C",  # Integrated Physics & Interactions
    ]

    # Add lookahead features
    features += [f"u_in_lead{i}" for i in range(1, Config.LEAD_STEPS + 1)]

    # Filter DataFrame
    # We also need to keep 'pressure' if it exists, and 'id'/'breath_id' for reshaping
    cols_to_keep = ["id", "breath_id", "u_out"] + features
    if "pressure" in df.columns:
        cols_to_keep.append("pressure")

    # Remove duplicates in list if any (u_out is in features and cols_to_keep)
    cols_to_keep = list(dict.fromkeys(cols_to_keep))

    return df[cols_to_keep], features


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches the data. Returns DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        train_loader, val_loader, test_loader, input_shape
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "train_u_out": os.path.join(cache_dir, "train_u_out.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "val_u_out": os.path.join(cache_dir, "val_u_out.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "test_u_out": os.path.join(cache_dir, "test_u_out.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in paths.values()) and os.path.exists(
        Config.SCALER_PATH
    )

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_x = np.load(paths["train_x"])
        train_y = np.load(paths["train_y"])
        train_u_out = np.load(paths["train_u_out"])

        val_x = np.load(paths["val_x"])
        val_y = np.load(paths["val_y"])
        val_u_out = np.load(paths["val_u_out"])

        test_x = np.load(paths["test_x"])
        test_ids = np.load(paths["test_ids"])
        test_u_out = np.load(paths["test_u_out"])

    else:
        print("Processing data from scratch...")

        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug Mode
        if Config.DEBUG:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
            train_breaths = train_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
            val_breaths = val_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
            test_breaths = test_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]

            train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

        # Feature Engineering
        print("Generating features...")
        train_df, feature_cols = add_features(train_df)
        val_df, _ = add_features(val_df)
        test_df, _ = add_features(test_df)

        # Scaling
        # We scale only the feature columns. u_out is binary, but RobustScaler handles it fine (median=0 or 1).
        # However, usually we don't scale binary flags.
        # But 'u_out' is listed in features in Idea 25 description.
        # We will scale all features defined in add_features.
        print("Fitting Scaler...")
        scaler = RobustScaler()

        # Fit on Train
        scaler.fit(train_df[feature_cols])

        # Transform
        train_x_flat = scaler.transform(train_df[feature_cols])
        val_x_flat = scaler.transform(val_df[feature_cols])
        test_x_flat = scaler.transform(test_df[feature_cols])

        # Save Scaler
        joblib.dump(scaler, Config.SCALER_PATH)

        # Reshaping to (N_breaths, 80, N_features)
        # We assume 80 steps per breath.
        print("Reshaping tensors...")

        def reshape_data(df, x_flat, is_test=False):
            # Number of breaths
            n_breaths = len(df) // Config.SEQ_LEN

            # Reshape Features
            x = x_flat.reshape(n_breaths, Config.SEQ_LEN, -1)

            # Reshape u_out (raw, for masking)
            u_out = df["u_out"].values.reshape(n_breaths, Config.SEQ_LEN)

            if not is_test:
                y = df["pressure"].values.reshape(n_breaths, Config.SEQ_LEN)
                return x, y, u_out, None
            else:
                ids = df["id"].values
                return x, None, u_out, ids

        train_x, train_y, train_u_out, _ = reshape_data(train_df, train_x_flat)
        val_x, val_y, val_u_out, _ = reshape_data(val_df, val_x_flat)
        test_x, _, test_u_out, test_ids = reshape_data(
            test_df, test_x_flat, is_test=True
        )

        # Cache Data
        print("Caching data...")
        np.save(paths["train_x"], train_x)
        np.save(paths["train_y"], train_y)
        np.save(paths["train_u_out"], train_u_out)

        np.save(paths["val_x"], val_x)
        np.save(paths["val_y"], val_y)
        np.save(paths["val_u_out"], val_u_out)

        np.save(paths["test_x"], test_x)
        np.save(paths["test_ids"], test_ids)
        np.save(paths["test_u_out"], test_u_out)

    # Create Datasets
    print("Creating Datasets...")
    train_dataset = VentilatorDataset(train_x, train_u_out, train_y)
    val_dataset = VentilatorDataset(val_x, val_u_out, val_y)
    test_dataset = VentilatorDataset(test_x, test_u_out, ids=test_ids, is_test=True)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    input_shape = train_x.shape[2]
    print(f"Data ready. Input shape: {train_x.shape}. Features: {input_shape}")

    return train_loader, val_loader, test_loader, input_shape

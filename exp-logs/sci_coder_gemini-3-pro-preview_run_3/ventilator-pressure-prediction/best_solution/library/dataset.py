import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import joblib
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, y=None, is_test=False):
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (80, n_features)
        x_sample = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            return x_sample
        else:
            # y shape: (80,)
            y_sample = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_sample, y_sample


def add_features(df):
    """
    Adds physics-based features and lookaheads.
    Strictly respects breath_id boundaries.
    """
    # Ensure data is sorted (it should be, but safety first)
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # 1. Time Delta (dt)
    # We calculate difference in time_step, but must mask out the jump between breaths
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume (Area) via Numerical Integration
    # area = integral(u_in * dt)
    df["area"] = (
        df.groupby("breath_id")
        .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
        .reset_index(level=0, drop=True)
    )

    # 3. u_in Derivative (Acceleration)
    df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # 4. Lookahead features (t+1 to t+4)
    # Shift negative means looking into the future
    for i in range(1, Config.N_LAGS + 1):
        df[f"u_in_lead{i}"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0)

    # 5. Explicit Forward Derivatives (Cite solution_lesson_node_00052)
    # Capturing the rate of change for future steps helps anticipate dynamics
    df["u_in_diff_next1"] = df["u_in_lead1"] - df["u_in"]
    df["u_in_diff_next2"] = df["u_in_lead2"] - df["u_in_lead1"]

    # 6. Interaction Terms
    df["R_uin"] = df["R"] * df["u_in"]
    # Avoid division by zero if C is 0 (unlikely in this dataset but good practice)
    df["area_C"] = df["area"] / df["C"]

    # Note: We do NOT drop columns here yet, we just select what we need later
    # based on Config.CONT_FEATURES

    return df


def prepare_datasets(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and reshaping.
    Implements caching mechanism using .npy files.
    """
    # Cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "scaler": Config.SCALER_PATH,
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        train_x = np.load(files["train_x"])
        train_y = np.load(files["train_y"])
        val_x = np.load(files["val_x"])
        val_y = np.load(files["val_y"])
        test_x = np.load(files["test_x"])
        # We don't necessarily return test_ids here, but it's good to know it's cached
        return train_x, train_y, val_x, val_y, test_x

    print("Cache not found or ignored. Processing data from scratch...")

    # 1. Load Data
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print("DEBUG Mode: Using subset of data")
        train_df = train_df.iloc[: 80 * 100]  # 100 breaths
        val_df = val_df.iloc[: 80 * 50]
        test_df = test_df.iloc[: 80 * 50]

    # 2. Feature Engineering
    print("Applying feature engineering...")
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # 3. Scaling
    print("Fitting Scaler...")
    feature_cols = Config.CONT_FEATURES
    scaler = RobustScaler()

    # Fit only on training data
    scaler.fit(train_df[feature_cols])

    # Transform all sets
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Save scaler
    joblib.dump(scaler, files["scaler"])

    # 4. Reshaping to (N_breaths, 80, N_features)
    print("Reshaping data...")

    # Each breath has exactly 80 time steps
    steps_per_breath = 80

    # Helper to reshape
    def reshape_data(df, is_test=False):
        # Ensure correct features are selected
        X = df[feature_cols].values
        num_breaths = len(df) // steps_per_breath

        # Reshape X: (N, 80, F)
        X = X.reshape(num_breaths, steps_per_breath, len(feature_cols))

        if is_test:
            return X, df["id"].values
        else:
            y = df[Config.TARGET_COL].values
            # Reshape y: (N, 80)
            y = y.reshape(num_breaths, steps_per_breath)
            return X, y

    train_x, train_y = reshape_data(train_df)
    val_x, val_y = reshape_data(val_df)
    test_x, test_ids = reshape_data(test_df, is_test=True)

    # 5. Save to Cache
    print("Saving to cache...")
    np.save(files["train_x"], train_x)
    np.save(files["train_y"], train_y)
    np.save(files["val_x"], val_x)
    np.save(files["val_y"], val_y)
    np.save(files["test_x"], test_x)
    np.save(files["test_ids"], test_ids)

    print(f"Data processing complete. Train shape: {train_x.shape}")

    return train_x, train_y, val_x, val_y, test_x

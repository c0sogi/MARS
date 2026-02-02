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
    Holds data in memory as numpy arrays and returns tensors.
    """

    def __init__(self, X, y=None, u_out=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, 80, F).
            y (np.ndarray, optional): Target pressure of shape (N, 80).
            u_out (np.ndarray, optional): Expiratory valve status of shape (N, 80).
        """
        self.X = X
        self.y = y
        self.u_out = u_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"input": torch.tensor(self.X[idx], dtype=torch.float32)}

        if self.y is not None:
            data["target"] = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.u_out is not None:
            data["u_out"] = torch.tensor(self.u_out[idx], dtype=torch.float32)

        return data


def add_features(df):
    """
    Adds physics-based and lag features to the dataframe.
    Uses vectorized numpy operations for efficiency.
    """
    # The dataset is structured as 80 steps per breath.
    # We reshape to (N_breaths, 80) to perform breath-wise operations.
    num_breaths = len(df) // 80

    # Extract columns as (N, 80) arrays
    u_in = df["u_in"].values.reshape(num_breaths, 80)
    time_step = df["time_step"].values.reshape(num_breaths, 80)
    R = df["R"].values.reshape(num_breaths, 80)
    C = df["C"].values.reshape(num_breaths, 80)

    # --- Feature Engineering ---

    # 1. dt (Time delta)
    # Calculate difference between steps. For the first step, we assume dt=0.
    dt = np.zeros_like(time_step)
    dt[:, 1:] = time_step[:, 1:] - time_step[:, :-1]

    # 2. u_in_diff (Acceleration/Derivative)
    u_in_diff = np.zeros_like(u_in)
    u_in_diff[:, 1:] = u_in[:, 1:] - u_in[:, :-1]

    # 3. Area (Volume) = Integral(u_in * dt)
    # Cumulative sum along the time axis
    area = np.cumsum(u_in * dt, axis=1)

    # 4. Interactions
    R_u_in = R * u_in
    area_C = area / C

    # 5. Lookahead Features (u_in_next_k)
    # Shift u_in backwards (future into present). Fill end with 0.
    lookaheads = {}
    for k in range(1, Config.LOOKAHEAD_STEPS + 1):
        shifted = np.roll(u_in, -k, axis=1)
        # The roll wraps around, so we must mask the last k elements
        shifted[:, -k:] = 0
        lookaheads[f"u_in_next_{k}"] = shifted

    # --- Assign back to DataFrame ---
    # Flatten arrays back to (Total_Rows,)
    df["dt"] = dt.flatten()
    df["u_in_diff"] = u_in_diff.flatten()
    df["area"] = area.flatten()
    df["R_u_in"] = R_u_in.flatten()
    df["area_C"] = area_C.flatten()

    for k, v in lookaheads.items():
        df[k] = v.flatten()

    return df


def load_and_preprocess(load_cached_data=True):
    """
    Main data pipeline function.
    Loads data, engineers features, scales, caches results, and returns Datasets.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_x": os.path.join(Config.CACHE_DIR, "train_x.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "train_u_out": os.path.join(Config.CACHE_DIR, "train_u_out.npy"),
        "val_x": os.path.join(Config.CACHE_DIR, "val_x.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "val_u_out": os.path.join(Config.CACHE_DIR, "val_u_out.npy"),
        "test_x": os.path.join(Config.CACHE_DIR, "test_x.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        "test_u_out": os.path.join(Config.CACHE_DIR, "test_u_out.npy"),
    }

    # Check if cache is valid
    cache_exists = all(
        os.path.exists(p) for p in cache_files.values()
    ) and os.path.exists(Config.SCALER_PATH)

    if load_cached_data and cache_exists:
        print("Loading cached data from", Config.CACHE_DIR)
        try:
            train_x = np.load(cache_files["train_x"])
            train_y = np.load(cache_files["train_y"])
            train_u_out = np.load(cache_files["train_u_out"])

            val_x = np.load(cache_files["val_x"])
            val_y = np.load(cache_files["val_y"])
            val_u_out = np.load(cache_files["val_u_out"])

            test_x = np.load(cache_files["test_x"])
            test_u_out = np.load(cache_files["test_u_out"])

            return (
                VentilatorDataset(train_x, train_y, train_u_out),
                VentilatorDataset(val_x, val_y, val_u_out),
                VentilatorDataset(test_x, None, test_u_out),
            )
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # --- Compute from Scratch ---
    print("Processing data from scratch...")

    # 1. Load Raw Data
    print(f"Loading CSVs from {Config.METADATA_DIR}...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print("DEBUG Mode: Using subset (1000 breaths)")
        train_df = train_df.iloc[: 80 * 1000]
        val_df = val_df.iloc[: 80 * 1000]
        test_df = test_df.iloc[: 80 * 1000]

    # 2. Feature Engineering
    print("Generating features...")
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # 3. Scaling
    print("Fitting RobustScaler...")
    feature_cols = Config.INPUT_FEATURES
    scaler = RobustScaler()

    # Flatten to (N_samples, N_features) for sklearn
    X_train_flat = train_df[feature_cols].values
    X_val_flat = val_df[feature_cols].values
    X_test_flat = test_df[feature_cols].values

    # Fit on Train, Transform all
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Save Scaler
    joblib.dump(scaler, Config.SCALER_PATH)

    # 4. Reshape to Sequences (N_breaths, 80, N_features)
    n_train = len(train_df) // 80
    n_val = len(val_df) // 80
    n_test = len(test_df) // 80

    train_x = X_train_scaled.reshape(n_train, 80, len(feature_cols))
    val_x = X_val_scaled.reshape(n_val, 80, len(feature_cols))
    test_x = X_test_scaled.reshape(n_test, 80, len(feature_cols))

    # 5. Extract Targets and Aux
    train_y = train_df[Config.TARGET_COL].values.reshape(n_train, 80)
    val_y = val_df[Config.TARGET_COL].values.reshape(n_val, 80)

    train_u_out = train_df["u_out"].values.reshape(n_train, 80)
    val_u_out = val_df["u_out"].values.reshape(n_val, 80)
    test_u_out = test_df["u_out"].values.reshape(n_test, 80)

    test_ids = test_df["id"].values  # Keep flat for submission generation

    # 6. Save to Cache
    print("Saving to cache...")
    np.save(cache_files["train_x"], train_x)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["train_u_out"], train_u_out)

    np.save(cache_files["val_x"], val_x)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["val_u_out"], val_u_out)

    np.save(cache_files["test_x"], test_x)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["test_u_out"], test_u_out)

    print("Data processing complete.")

    return (
        VentilatorDataset(train_x, train_y, train_u_out),
        VentilatorDataset(val_x, val_y, val_u_out),
        VentilatorDataset(test_x, None, test_u_out),
    )

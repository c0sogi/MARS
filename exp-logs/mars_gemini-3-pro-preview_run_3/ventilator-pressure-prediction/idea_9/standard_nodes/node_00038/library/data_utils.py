import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class RobustScaler:
    """
    Custom Robust Scaler using Median and IQR.
    Scales features using statistics computed from the training set.
    Robust to outliers which is critical for valve control inputs.
    """

    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        # X shape: (N_samples, Seq_len, N_features)
        # Compute stats across samples and time steps
        self.median = np.nanmedian(X, axis=(0, 1))
        q25 = np.nanpercentile(X, 25, axis=(0, 1))
        q75 = np.nanpercentile(X, 75, axis=(0, 1))
        self.iqr = q75 - q25

        # Handle constant features (IQR=0) to prevent division by zero
        self.iqr[self.iqr == 0] = 1.0

    def transform(self, X):
        if self.median is None or self.iqr is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.median) / self.iqr

    def save(self, path):
        np.savez(path, median=self.median, iqr=self.iqr)

    def load(self, path):
        data = np.load(path)
        self.median = data["median"]
        self.iqr = data["iqr"]


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            self.u_out = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx]}
        if self.y is not None:
            data["y"] = self.y[idx]
        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]
        return data


def compute_features(df):
    """
    Computes features for the RPC-Net pipeline.
    Reshapes flat DataFrame to (N_breaths, 80, N_features).
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"])

    # Reshape to (N_breaths, 80, N_cols)
    # We assume standard 80 steps per breath
    n_breaths = len(df) // Config.SEQ_LEN

    # Extract raw columns as numpy arrays for vectorized calc
    # Columns in metadata: id, breath_id, R, C, time_step, u_in, u_out, [pressure]

    # Helper to get reshaped array for a column
    def get_col(col_name):
        return df[col_name].values.reshape(n_breaths, Config.SEQ_LEN)

    # Raw Features
    u_in = get_col("u_in")
    u_out = get_col("u_out")
    R = get_col("R")
    C = get_col("C")
    time_step = get_col("time_step")

    # Target (if exists)
    pressure = None
    if "pressure" in df.columns:
        pressure = get_col("pressure")
        # Reshape pressure to (N, 80, 1)
        pressure = pressure[:, :, np.newaxis]

    # --- Feature Engineering ---

    # 1. Dynamic PID Terms
    # Integral (Volume proxy)
    # Using simple cumsum. For more physical accuracy, could use trapezoidal with time_step,
    # but cumsum is often sufficient for NN to learn.
    u_in_cumsum = np.cumsum(u_in, axis=1)

    # Derivative (Flow Acceleration proxy)
    # Pad with 0 at the beginning
    u_in_diff1 = np.diff(u_in, axis=1, prepend=0)

    # Second Derivative
    u_in_diff2 = np.diff(u_in_diff1, axis=1, prepend=0)

    # 2. Contextual Lags/Leads
    # Shift along time axis (axis 1)
    def shift(arr, num, fill_value=0):
        result = np.empty_like(arr)
        if num > 0:
            result[:, :num] = fill_value
            result[:, num:] = arr[:, :-num]
        elif num < 0:
            result[:, num:] = fill_value
            result[:, :num] = arr[:, -num:]
        else:
            result[:] = arr
        return result

    u_in_lag1 = shift(u_in, 1)
    u_in_lag2 = shift(u_in, 2)
    u_in_lead1 = shift(u_in, -1)
    u_in_lead2 = shift(u_in, -2)

    # 3. Physics Interactions
    R_u_in = R * u_in
    # Avoid division by zero if C is 0 (though C is usually 10, 20, 50)
    vol_C_ratio = u_in_cumsum / (C + 1e-5)

    # --- Feature Assembly ---
    # Must match Config.FEATURE_LIST order
    # FEATURE_LIST = [
    #     "time_step", "u_in", "u_out", "R", "C",
    #     "u_in_cumsum", "u_in_diff1", "u_in_diff2",
    #     "u_in_lag1", "u_in_lag2", "u_in_lead1", "u_in_lead2",
    #     "R_u_in", "vol_C_ratio"
    # ]

    feature_map = {
        "time_step": time_step,
        "u_in": u_in,
        "u_out": u_out,
        "R": R,
        "C": C,
        "u_in_cumsum": u_in_cumsum,
        "u_in_diff1": u_in_diff1,
        "u_in_diff2": u_in_diff2,
        "u_in_lag1": u_in_lag1,
        "u_in_lag2": u_in_lag2,
        "u_in_lead1": u_in_lead1,
        "u_in_lead2": u_in_lead2,
        "R_u_in": R_u_in,
        "vol_C_ratio": vol_C_ratio,
    }

    features_list = []
    for fname in Config.FEATURE_LIST:
        if fname in feature_map:
            features_list.append(feature_map[fname][:, :, np.newaxis])
        else:
            raise ValueError(f"Feature {fname} not implemented in compute_features")

    X = np.concatenate(features_list, axis=2)

    # Return u_out separately for masking loss
    # Reshape u_out to (N, 80, 1)
    u_out_reshaped = u_out[:, :, np.newaxis]

    return X, pressure, u_out_reshaped


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main data pipeline entry point.
    Handles loading, preprocessing, caching, scaling, and DataLoader creation.
    """
    set_seed(Config.SEED)
    Config.create_dirs()

    # Check if we need to recompute
    cache_exists = (
        os.path.exists(Config.TRAIN_X_CACHE)
        and os.path.exists(Config.TRAIN_Y_CACHE)
        and os.path.exists(Config.TRAIN_U_OUT_CACHE)
        and os.path.exists(Config.VAL_X_CACHE)
        and os.path.exists(Config.VAL_Y_CACHE)
        and os.path.exists(Config.VAL_U_OUT_CACHE)
        and os.path.exists(Config.TEST_X_CACHE)
        and os.path.exists(Config.TEST_U_OUT_CACHE)
        and os.path.exists(Config.SCALER_CACHE)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from working directory...")
        train_x = np.load(Config.TRAIN_X_CACHE)
        train_y = np.load(Config.TRAIN_Y_CACHE)
        train_u_out = np.load(Config.TRAIN_U_OUT_CACHE)

        val_x = np.load(Config.VAL_X_CACHE)
        val_y = np.load(Config.VAL_Y_CACHE)
        val_u_out = np.load(Config.VAL_U_OUT_CACHE)

        test_x = np.load(Config.TEST_X_CACHE)
        test_u_out = np.load(Config.TEST_U_OUT_CACHE)

    else:
        print("Processing data from scratch...")

        # Load raw CSVs
        print(f"Loading {Config.TRAIN_CSV}...")
        train_df = pd.read_csv(Config.TRAIN_CSV)
        print(f"Loading {Config.VAL_CSV}...")
        val_df = pd.read_csv(Config.VAL_CSV)
        print(f"Loading {Config.TEST_CSV}...")
        test_df = pd.read_csv(Config.TEST_CSV)

        if debug:
            print("Debug mode: subsampling data...")
            # Take first 100 breaths (100 * 80 rows)
            train_df = train_df.iloc[: 100 * Config.SEQ_LEN]
            val_df = val_df.iloc[: 100 * Config.SEQ_LEN]
            test_df = test_df.iloc[: 100 * Config.SEQ_LEN]

        # Compute Features
        print("Computing features for Training set...")
        train_x, train_y, train_u_out = compute_features(train_df)

        print("Computing features for Validation set...")
        val_x, val_y, val_u_out = compute_features(val_df)

        print("Computing features for Test set...")
        test_x, _, test_u_out = compute_features(test_df)

        # Scaling
        if Config.USE_ROBUST_SCALER:
            print("Fitting RobustScaler...")
            scaler = RobustScaler()
            scaler.fit(train_x)
            scaler.save(Config.SCALER_CACHE)

            print("Transforming data...")
            train_x = scaler.transform(train_x)
            val_x = scaler.transform(val_x)
            test_x = scaler.transform(test_x)

        # Save to Cache
        print("Saving data to cache...")
        np.save(Config.TRAIN_X_CACHE, train_x)
        np.save(Config.TRAIN_Y_CACHE, train_y)
        np.save(Config.TRAIN_U_OUT_CACHE, train_u_out)

        np.save(Config.VAL_X_CACHE, val_x)
        np.save(Config.VAL_Y_CACHE, val_y)
        np.save(Config.VAL_U_OUT_CACHE, val_u_out)

        np.save(Config.TEST_X_CACHE, test_x)
        np.save(Config.TEST_U_OUT_CACHE, test_u_out)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_x, val_y, val_u_out)
    test_dataset = VentilatorDataset(test_x, u_out=test_u_out)  # No target for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader

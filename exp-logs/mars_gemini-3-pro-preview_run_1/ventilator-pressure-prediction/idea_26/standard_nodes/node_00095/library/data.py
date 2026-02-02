import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Dataset Class
# =========================================================================


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves (features, u_out_mask, target).
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Feature array of shape (N, 80, F).
            u_out (np.ndarray): Binary mask array of shape (N, 80) for loss masking.
            y (np.ndarray, optional): Target pressure array of shape (N, 80).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.u_out[idx], self.y[idx]
        return self.X[idx], self.u_out[idx]


# =========================================================================
# Feature Engineering
# =========================================================================


def engineer_features(df):
    """
    Applies physics-based feature engineering, lags, and differences.
    """
    # Ensure sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # 1. Physics: Volume Integration
    # volume = sum(u_in * dt)
    # Calculate dt
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # 2. Physics: Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    # Avoid division by zero if C is 0 (physically unlikely but safe to handle)
    df["vol_C"] = df["volume"] / df["C"]

    # 3. Dynamics: Lags
    # We use groupby shift to ensure no leakage between breaths
    for lag in range(1, Config.LAG_STEPS + 1):
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # 4. Dynamics: Differences
    # First difference
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0)
    # Second difference
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff(1).fillna(0)

    # Drop temporary columns
    df = df.drop(columns=["dt"])

    return df


# =========================================================================
# Data Loading and Processing
# =========================================================================


def load_and_preprocess_dataframe(csv_path, cache_path, is_train=False):
    """
    Loads data from CSV or Cache, applies engineering, and returns DataFrame.
    """
    # Check cache first (only if not in DEBUG mode to avoid polluting cache with partial data)
    if not Config.DEBUG and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Loading raw data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Debug Mode: Slice data
        if Config.DEBUG:
            print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLES} breaths...")
            breath_ids = df["breath_id"].unique()[: Config.DEBUG_SAMPLES]
            df = df[df["breath_id"].isin(breath_ids)].copy()

        # Feature Engineering
        print("Engineering features...")
        df = engineer_features(df)

        # Cache result (only if not debugging)
        if not Config.DEBUG:
            print(f"Saving engineered data to {cache_path}...")
            df.to_parquet(cache_path, index=False)

    return df


def get_feature_columns(df):
    """
    Returns the list of feature columns to be used for training.
    """
    # Exclude non-feature columns
    exclude = ["id", "breath_id", "pressure", "u_out_raw"]
    features = [c for c in df.columns if c not in exclude]
    return features


def fit_and_save_scaler(df, features):
    """
    Fits RobustScaler on training features and saves statistics to NPY.
    """
    print("Fitting RobustScaler on training data...")
    scaler = RobustScaler()
    scaler.fit(df[features].values)

    # Save statistics
    center_path, scale_path = Config.get_scaler_paths()
    np.save(center_path, scaler.center_)
    np.save(scale_path, scaler.scale_)
    print(f"Scaler statistics saved to {center_path} and {scale_path}")

    return scaler


def load_scaler():
    """
    Loads RobustScaler from saved NPY statistics.
    """
    center_path, scale_path = Config.get_scaler_paths()
    if not os.path.exists(center_path) or not os.path.exists(scale_path):
        raise FileNotFoundError(
            "Scaler statistics not found. Run training preparation first."
        )

    center = np.load(center_path)
    scale = np.load(scale_path)

    scaler = RobustScaler()
    scaler.center_ = center
    scaler.scale_ = scale
    return scaler


def reshape_to_sequences(df, features, target_col="pressure"):
    """
    Reshapes DataFrame to (N_breaths, 80, N_features).
    """
    # Ensure strict sorting
    df = df.sort_values(["breath_id", "time_step"])

    # Extract arrays
    X = df[features].values
    u_out = df["u_out"].values

    # Calculate number of breaths
    num_breaths = len(df) // Config.SEQ_LEN

    # Reshape
    # X: (N, 80, F)
    X = X.reshape(num_breaths, Config.SEQ_LEN, -1)
    # u_out: (N, 80)
    u_out = u_out.reshape(num_breaths, Config.SEQ_LEN)

    if target_col in df.columns:
        y = df[target_col].values
        y = y.reshape(num_breaths, Config.SEQ_LEN)
        return X, u_out, y

    return X, u_out, None


# =========================================================================
# Main Interface
# =========================================================================


def prepare_data():
    """
    Main function to prepare DataLoaders for Train, Val, and Test.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    seed_everything()

    # 1. Load DataFrames
    train_df = load_and_preprocess_dataframe(
        Config.TRAIN_PATH, Config.TRAIN_CACHE_PATH, is_train=True
    )
    val_df = load_and_preprocess_dataframe(Config.VAL_PATH, Config.VAL_CACHE_PATH)
    test_df = load_and_preprocess_dataframe(Config.TEST_PATH, Config.TEST_CACHE_PATH)

    # 2. Identify Features
    features = get_feature_columns(train_df)
    print(f"Selected {len(features)} features: {features}")

    # 3. Scaling
    # Fit on Train, Apply to All
    # Check if scaler stats exist, if not fit, else load (or just refit if training)
    if not os.path.exists(Config.SCALER_CENTER_PATH) or Config.DEBUG:
        scaler = fit_and_save_scaler(train_df, features)
    else:
        print("Loading existing scaler...")
        scaler = load_scaler()

    print("Scaling data...")
    train_df[features] = scaler.transform(train_df[features].values)
    val_df[features] = scaler.transform(val_df[features].values)
    test_df[features] = scaler.transform(test_df[features].values)

    # 4. Reshape
    print("Reshaping to sequences...")
    X_train, u_out_train, y_train = reshape_to_sequences(train_df, features)
    X_val, u_out_val, y_val = reshape_to_sequences(val_df, features)
    X_test, u_out_test, _ = reshape_to_sequences(test_df, features)

    # Extract Test IDs for submission (flattened)
    # We need the IDs corresponding to the test sequences
    # Since reshape assumes sorted order, we take the IDs and reshape/flatten logic consistent
    test_ids = test_df["id"].values

    # 5. Create Datasets
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    test_dataset = VentilatorDataset(X_test, u_out_test, None)

    print(f"Train Dataset: {len(train_dataset)} breaths")
    print(f"Val Dataset:   {len(val_dataset)} breaths")
    print(f"Test Dataset:  {len(test_dataset)} breaths")

    # 6. Create Loaders
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

    return train_loader, val_loader, test_loader, test_ids

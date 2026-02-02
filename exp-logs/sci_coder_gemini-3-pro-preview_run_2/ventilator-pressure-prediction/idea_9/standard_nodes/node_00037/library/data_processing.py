import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

from library.config import Config
from library.utils import seed_everything


# ==========================================
# Dataset Class
# ==========================================
class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves data in the shape (Batch, Time_Steps, Features).
    """

    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features of shape (Num_Breaths, 80, Num_Features).
            y (np.ndarray, optional): Target pressure of shape (Num_Breaths, 80).
            is_test (bool): Flag indicating if this is the test set.
        """
        self.X = X
        self.y = y
        self.is_test = is_test

        # Basic validation
        if not self.is_test and self.y is not None:
            assert self.X.shape[0] == self.y.shape[0], "Mismatch in X and y dimensions"
            assert self.X.shape[1] == self.y.shape[1], "Mismatch in time steps"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensor (Float32 is standard for this task)
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test or self.y is None:
            return x_tensor

        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_tensor, y_tensor


# ==========================================
# Feature Engineering
# ==========================================
def engineer_features(df):
    """
    Applies physics-based feature engineering to the dataframe.
    Calculates integrals, derivatives, and interaction terms.
    """
    # Ensure sorted by breath and time for correct diff/shift
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # 1. Time Delta and Volume Integration
    # dt = current_time - prev_time. Fill NaN (first step) with 0.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Volume = Integral(Flow * dt). u_in is a proxy for flow.
    # We calculate the incremental volume then cumsum.
    df["volume_inc"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["volume_inc"].cumsum()

    # 2. Physics Interaction Terms
    # Resistive Pressure ~ R * Flow
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure ~ Volume / Compliance
    df["u_in_cumsum_C"] = df["u_in_cumsum"] / df["C"]

    # 3. Dynamics (Lags)
    # Provide history of control input to the model
    for lag in Config.LAG_STEPS:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # 4. Dynamics (Finite Differences / Derivatives)
    # Provide velocity and acceleration of control input
    # Config.DIFF_STEPS usually [1, 2]
    for diff in Config.DIFF_STEPS:
        df[f"u_in_diff{diff}"] = df.groupby("breath_id")["u_in"].diff(diff).fillna(0)

    # Explicitly add diff3 and diff4 as requested in SELECTED_FEATURES
    df["u_in_diff3"] = df.groupby("breath_id")["u_in"].diff(3).fillna(0)
    df["u_in_diff4"] = df.groupby("breath_id")["u_in"].diff(4).fillna(0)

    # Cleanup temporary columns
    df = df.drop(columns=["dt", "volume_inc"], errors="ignore")

    # Fill any remaining NaNs (e.g., from lags/diffs at start of breath)
    df = df.fillna(0)

    return df


# ==========================================
# Data Processing Pipeline
# ==========================================
def process_partition(df_raw, scaler=None, fit_scaler=False):
    """
    Engineers features, scales data, and formats for the model.
    """
    # Feature Engineering
    df = engineer_features(df_raw)

    # Scaling
    # We only scale continuous features, not categorical (u_out) or targets
    if fit_scaler:
        scaler = RobustScaler()
        df[Config.CONT_FEATURES] = scaler.fit_transform(df[Config.CONT_FEATURES])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided if fit_scaler=False")
        df[Config.CONT_FEATURES] = scaler.transform(df[Config.CONT_FEATURES])

    return df, scaler


def reshape_to_sequences(df, features, target_col="pressure"):
    """
    Reshapes flat dataframe to (N_breaths, 80, N_features).
    Assumes data is sorted by breath_id and time_step.
    """
    # Verify assumption: 80 steps per breath
    # In this dataset, breaths are typically fixed length.
    # We calculate steps per breath to be safe or enforce 80.
    steps_per_breath = 80
    num_breaths = len(df) // steps_per_breath

    if len(df) % steps_per_breath != 0:
        # Fallback: If dataset size isn't perfect multiple (e.g. debugging subset), truncate
        print(
            f"Warning: Data length {len(df)} is not a multiple of {steps_per_breath}. Truncating."
        )
        num_breaths = len(df) // steps_per_breath
        df = df.iloc[: num_breaths * steps_per_breath]

    # Extract Feature Matrix
    X = df[features].values.reshape(num_breaths, steps_per_breath, len(features))

    # Extract Target Matrix (if exists)
    y = None
    if target_col in df.columns:
        y = df[target_col].values.reshape(num_breaths, steps_per_breath)

    return X, y


def prepare_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Main entry point. Loads data, processes/caches it, and returns DataLoaders.
    """
    seed_everything()

    # 1. Check Cache
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE)
        and os.path.exists(Config.VAL_CACHE)
        and os.path.exists(Config.TEST_CACHE)
        and os.path.exists(Config.SCALER_CACHE)
    )

    if Config.USE_CACHE and cache_exists:
        print("Loading processed data from cache...")
        df_train = pd.read_parquet(Config.TRAIN_CACHE)
        df_val = pd.read_parquet(Config.VAL_CACHE)
        df_test = pd.read_parquet(Config.TEST_CACHE)
        scaler = joblib.load(Config.SCALER_CACHE)
    else:
        print("Processing data from scratch...")

        # Load Raw Data
        train_raw = pd.read_csv(Config.TRAIN_CSV)
        test_raw = pd.read_csv(Config.TEST_CSV)

        # Load Metadata for Splitting
        train_meta = pd.read_csv(Config.TRAIN_META)
        val_meta = pd.read_csv(Config.VAL_META)

        # Identify breath_ids for split
        train_breath_ids = train_meta["breath_id"].unique()
        val_breath_ids = val_meta["breath_id"].unique()

        # Split Train/Val
        # Note: train_raw contains both train and val breaths
        df_train_raw = train_raw[train_raw["breath_id"].isin(train_breath_ids)].copy()
        df_val_raw = train_raw[train_raw["breath_id"].isin(val_breath_ids)].copy()

        if debug:
            print("Debug mode: Using subset of data")
            # Take first 100 breaths for each
            train_subset_ids = df_train_raw["breath_id"].unique()[:100]
            val_subset_ids = df_val_raw["breath_id"].unique()[:100]
            df_train_raw = df_train_raw[
                df_train_raw["breath_id"].isin(train_subset_ids)
            ].copy()
            df_val_raw = df_val_raw[df_val_raw["breath_id"].isin(val_subset_ids)].copy()
            test_subset_ids = test_raw["breath_id"].unique()[:100]
            test_raw = test_raw[test_raw["breath_id"].isin(test_subset_ids)].copy()

        # Process Train (Fit Scaler)
        print("Engineering features for Training set...")
        df_train, scaler = process_partition(df_train_raw, fit_scaler=True)

        # Process Val (Use Scaler)
        print("Engineering features for Validation set...")
        df_val, _ = process_partition(df_val_raw, scaler=scaler, fit_scaler=False)

        # Process Test (Use Scaler)
        print("Engineering features for Test set...")
        df_test, _ = process_partition(test_raw, scaler=scaler, fit_scaler=False)

        # Cache Results
        if not debug:
            print(f"Saving cache to {Config.WORKING_DIR}...")
            df_train.to_parquet(Config.TRAIN_CACHE)
            df_val.to_parquet(Config.VAL_CACHE)
            df_test.to_parquet(Config.TEST_CACHE)
            joblib.dump(scaler, Config.SCALER_CACHE)

        # Clean up raw data
        del train_raw, test_raw, df_train_raw, df_val_raw
        gc.collect()

    # 2. Reshape to Sequences
    print("Reshaping data to sequences (N, 80, F)...")
    # Ensure columns are in the correct order as defined in Config
    features = Config.SELECTED_FEATURES

    X_train, y_train = reshape_to_sequences(df_train, features)
    X_val, y_val = reshape_to_sequences(df_val, features)
    X_test, _ = reshape_to_sequences(df_test, features)

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # 3. Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test, is_test=True)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
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

    return train_loader, val_loader, test_loader

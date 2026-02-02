import os
import gc
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
    Serves sequences of shape (80, N_features).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, u_out: np.ndarray = None):
        self.X = X
        self.y = y
        self.u_out = u_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.X[idx], dtype=torch.float32),
        }
        if self.y is not None:
            data["target"] = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.u_out is not None:
            data["u_out"] = torch.tensor(self.u_out[idx], dtype=torch.float32)
        return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies physics-based feature engineering and temporal dynamics.
    """
    # Ensure sorted order for time-series operations
    df = df.sort_values(by=["breath_id", "time_step"]).reset_index(drop=True)

    # Time delta
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Time-weighted volume integration (Flow * dt)
    df["volume"] = (
        df.groupby("breath_id")
        .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
        .reset_index(level=0, drop=True)
    )

    # Physics Interaction Terms (Equation of Motion approximations)
    df["R_u_in"] = df["R"] * df["u_in"]  # Resistive component
    df["vol_C"] = df["volume"] / df["C"]  # Elastic component

    # Lag features (Temporal Context)
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1).fillna(0)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2).fillna(0)

    # Finite Differences (Dynamics)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - (df["u_in_lag1"] - df["u_in_lag2"])

    # Additional interactions
    df["area"] = df["time_step"] * df["u_in"]
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    # Fill any remaining NaNs
    df = df.fillna(0)

    return df


def prepare_data(config: Config, load_cached_data: bool = True):
    """
    Loads, processes, scales, and reshapes data.
    Handles caching to parquet files in config.WORKING_DIR.
    """

    # Check if cache exists
    cache_exists = (
        os.path.exists(config.TRAIN_CACHE)
        and os.path.exists(config.VAL_CACHE)
        and os.path.exists(config.TEST_CACHE)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {config.WORKING_DIR}...")
        train_df = pd.read_parquet(config.TRAIN_CACHE)
        val_df = pd.read_parquet(config.VAL_CACHE)
        test_df = pd.read_parquet(config.TEST_CACHE)

    else:
        print("Processing data from scratch...")
        seed_everything(config.SEED)

        # Load Raw Data
        print(f"Loading raw data from {config.INPUT_DIR}...")
        train_raw = pd.read_csv(config.TRAIN_CSV)
        test_raw = pd.read_csv(config.TEST_CSV)

        # Load Metadata for Splitting
        train_meta = pd.read_csv(config.TRAIN_META)
        val_meta = pd.read_csv(config.VAL_META)

        train_breath_ids = set(train_meta["breath_id"].unique())
        val_breath_ids = set(val_meta["breath_id"].unique())

        # Feature Engineering
        print("Engineering features...")
        train_raw = add_features(train_raw)
        test_raw = add_features(test_raw)

        # Split Train/Val
        print("Splitting train/val...")
        train_df = train_raw[train_raw["breath_id"].isin(train_breath_ids)].copy()
        val_df = train_raw[train_raw["breath_id"].isin(val_breath_ids)].copy()
        test_df = test_raw.copy()

        del train_raw, test_raw
        gc.collect()

        # Scaling
        # Define continuous columns to scale
        scale_cols = [
            "time_step",
            "u_in",
            "R",
            "C",
            "volume",
            "R_u_in",
            "vol_C",
            "u_in_lag1",
            "u_in_lag2",
            "u_in_lag_back1",
            "u_in_lag_back2",
            "u_in_diff1",
            "u_in_diff2",
            "area",
            "cross",
            "cross2",
            "dt",
        ]
        # Filter to ensure columns exist
        scale_cols = [c for c in scale_cols if c in train_df.columns]

        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        scaler.fit(train_df[scale_cols])

        # Save scaler params
        np.save(config.SCALER_CACHE, {"center": scaler.center_, "scale": scaler.scale_})

        print("Transforming all splits...")
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Debug Subsampling (Cite debug_lesson_7: Cache the Final State)
        if config.DEBUG:
            print("DEBUG Mode: Subsampling data before caching...")
            limit = 100
            # Each breath has 80 time steps
            limit_rows = limit * 80
            train_df = train_df.iloc[:limit_rows]
            val_df = val_df.iloc[:limit_rows]
            test_df = test_df.iloc[:limit_rows]

        # Save to Cache
        print(f"Saving processed data to {config.WORKING_DIR}...")
        train_df.to_parquet(config.TRAIN_CACHE)
        val_df.to_parquet(config.VAL_CACHE)
        test_df.to_parquet(config.TEST_CACHE)

    # --- Reshaping to 3D Tensors ---
    print("Reshaping data to (N, 80, Features)...")

    # Identify feature columns (exclude IDs and Targets)
    exclude_cols = ["id", "breath_id", "pressure"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Ensure u_out is included in features (it's a control input)
    if "u_out" not in feature_cols:
        feature_cols.append("u_out")

    feature_cols = sorted(feature_cols)
    print(f"Num features: {len(feature_cols)}")

    def to_3d(df, feats):
        # Assumes 80 steps per breath
        X = df[feats].values
        # Reshape: (Total_Rows / 80, 80, Num_Feats)
        X = X.reshape(-1, 80, len(feats))
        return X

    train_X = to_3d(train_df, feature_cols)
    val_X = to_3d(val_df, feature_cols)
    test_X = to_3d(test_df, feature_cols)

    train_y = train_df["pressure"].values.reshape(-1, 80)
    val_y = val_df["pressure"].values.reshape(-1, 80)

    # Extract u_out separately for Loss masking
    train_u_out = train_df["u_out"].values.reshape(-1, 80)
    val_u_out = val_df["u_out"].values.reshape(-1, 80)

    # Extract IDs for test submission mapping
    test_ids = test_df["id"].values.reshape(-1, 80)

    return (
        (train_X, train_y, train_u_out),
        (val_X, val_y, val_u_out),
        (test_X, test_ids),
    )


def get_dataloaders(config: Config):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    (train_X, train_y, train_u_out), (val_X, val_y, val_u_out), (test_X, test_ids) = (
        prepare_data(config, load_cached_data=True)
    )

    train_dataset = VentilatorDataset(train_X, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_X, val_y, val_u_out)
    test_dataset = VentilatorDataset(test_X)  # No target for test

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, test_ids

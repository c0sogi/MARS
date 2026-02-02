import os
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
    Reshapes flat data into sequences of length Config.SEQ_LEN (80).
    """

    def __init__(self, df, mode="train"):
        self.mode = mode
        self.seq_len = Config.SEQ_LEN

        # Ensure data is sorted by breath_id and time_step (id)
        # Assuming df is already sorted during processing

        # Extract features
        self.feature_cols = Config.FEATURE_COLS
        self.inputs = df[self.feature_cols].values.astype(np.float32)

        # Extract u_out for loss weighting
        if "u_out" in df.columns:
            self.u_out = df["u_out"].values.astype(np.float32)
        else:
            # Fallback if u_out not in columns (unlikely given config)
            self.u_out = np.zeros(len(df), dtype=np.float32)

        # Extract targets (pressure)
        if self.mode in ["train", "val"]:
            self.targets = df["pressure"].values.astype(np.float32)
        else:
            # For test set, dummy targets
            self.targets = np.zeros(len(df), dtype=np.float32)

        # Reshape to (num_breaths, seq_len, features)
        # We assume the dataframe length is perfectly divisible by seq_len
        # and ordered correctly.
        num_breaths = len(df) // self.seq_len

        self.inputs = self.inputs.reshape(num_breaths, self.seq_len, -1)
        self.u_out = self.u_out.reshape(num_breaths, self.seq_len)
        self.targets = self.targets.reshape(num_breaths, self.seq_len)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Returns: inputs, targets, u_out
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        w = torch.tensor(self.u_out[idx], dtype=torch.float32)
        return x, y, w


def add_features(df):
    """
    Implements Time-Weighted Physics Engineering.
    Calculates Volume, Interaction Terms, Lags, and Diffs.
    """
    # Ensure sorted order for correct lag/diff calculation
    df = df.sort_values(by=["breath_id", "id"]).reset_index(drop=True)

    # 1. Time Delta (dt)
    # Calculate difference in time_step grouped by breath_id
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume (Time-Weighted Integration)
    # u_in_cumsum = cumsum(u_in * dt)
    df["vol_increment"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["vol_increment"].cumsum()

    # 3. Physics Interaction Terms
    # R * u_in (Resistive Pressure proxy)
    df["R_u_in"] = df["R"] * df["u_in"]
    # Volume / C (Elastic Pressure proxy)
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # 4. Dynamics (Lags)
    # Using groupby shift to respect breath boundaries
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)

    # 5. Dynamics (Finite Differences)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]

    # Cleanup intermediate columns if necessary, but keeping them doesn't hurt
    # provided we only select Config.FEATURE_COLS later.

    # Fill any remaining NaNs (e.g., from lags) with 0
    df = df.fillna(0)

    return df


def prepare_datasets(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and caching.
    """
    seed_everything(Config.SEED)

    # Define cache paths
    cache_train = Config.CACHE_TRAIN_PATH
    cache_val = Config.CACHE_VAL_PATH
    cache_test = Config.CACHE_TEST_PATH
    scaler_path = Config.SCALER_PATH

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
        and os.path.exists(scaler_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached datasets...")
        df_train = pd.read_parquet(cache_train)
        df_val = pd.read_parquet(cache_val)
        df_test = pd.read_parquet(cache_test)

    else:
        print("Processing data from scratch...")

        # 1. Load Metadata to identify splits
        print("Loading metadata...")
        meta_train = pd.read_csv(Config.TRAIN_META)
        meta_val = pd.read_csv(Config.VAL_META)
        # meta_test = pd.read_csv(Config.TEST_META) # Not strictly needed for splitting test.csv

        train_breath_ids = set(meta_train["breath_id"].unique())
        val_breath_ids = set(meta_val["breath_id"].unique())

        # 2. Load Raw Data
        print("Loading raw csv files...")
        df_raw_train = pd.read_csv(Config.TRAIN_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # 3. Split Train/Val
        print("Splitting train/val...")
        df_train = df_raw_train[df_raw_train["breath_id"].isin(train_breath_ids)].copy()
        df_val = df_raw_train[df_raw_train["breath_id"].isin(val_breath_ids)].copy()

        del df_raw_train  # Free memory

        # 4. Feature Engineering
        print("Applying feature engineering...")
        df_train = add_features(df_train)
        df_val = add_features(df_val)
        df_test = add_features(df_test)

        # 5. Scaling
        print("Fitting scaler...")
        scaler = RobustScaler()

        # Select columns to scale.
        # Usually we scale continuous features. u_out is binary (0/1), better left alone or scaled?
        # RobustScaler handles outliers well.
        # We will scale all FEATURE_COLS.
        cols_to_scale = Config.FEATURE_COLS

        # Fit on Train
        scaler.fit(df_train[cols_to_scale])

        # Transform Train, Val, Test
        print("Transforming data...")
        df_train[cols_to_scale] = scaler.transform(df_train[cols_to_scale])
        df_val[cols_to_scale] = scaler.transform(df_val[cols_to_scale])
        df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

        # 6. Caching
        print(f"Saving cache to {Config.WORKING_DIR}...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        df_train.to_parquet(cache_train, index=False)
        df_val.to_parquet(cache_val, index=False)
        df_test.to_parquet(cache_test, index=False)

        # Save scaler params (center, scale)
        np.save(scaler_path, {"center": scaler.center_, "scale": scaler.scale_})

    # Debugging: Sample data if Config.DEBUG is True
    if Config.DEBUG:
        print("Debug mode: Sampling data...")
        sample_size = Config.get_sample_size()

        # Sample by breath_id to keep sequences intact
        train_breaths = df_train["breath_id"].unique()[:sample_size]
        val_breaths = df_val["breath_id"].unique()[:sample_size]
        test_breaths = df_test["breath_id"].unique()[:sample_size]

        df_train = df_train[df_train["breath_id"].isin(train_breaths)].copy()
        df_val = df_val[df_val["breath_id"].isin(val_breaths)].copy()
        df_test = df_test[df_test["breath_id"].isin(test_breaths)].copy()

        print(f"Debug Train shape: {df_train.shape}")

    return df_train, df_val, df_test


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    df_train, df_val, df_test = prepare_datasets(load_cached_data=load_cached_data)

    train_dataset = VentilatorDataset(df_train, mode="train")
    val_dataset = VentilatorDataset(df_val, mode="val")
    test_dataset = VentilatorDataset(df_test, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for stability in training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

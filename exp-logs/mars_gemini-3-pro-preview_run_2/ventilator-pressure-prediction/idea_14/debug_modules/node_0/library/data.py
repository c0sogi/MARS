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
    Groups flat dataframe rows into sequences of length 80 (per breath).
    """

    def __init__(self, df, mode="train"):
        self.mode = mode
        self.seq_len = Config.SEQ_LEN

        # Ensure the dataframe is sorted by breath_id and time_step/id
        # (This is expected to be handled during preprocessing)

        # Extract features as float32
        self.inputs = df[Config.INPUT_FEATURES].values.astype(np.float32)

        # Extract u_out for masking (1=expiratory, 0=inspiratory)
        self.u_outs = df["u_out"].values.astype(np.float32)

        # Extract targets if available
        if self.mode != "test":
            self.targets = df["pressure"].values.astype(np.float32)
        else:
            self.targets = None

        # Calculate number of breaths
        self.num_breaths = len(df) // self.seq_len

    def __len__(self):
        return self.num_breaths

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        # Get sequence features
        x = self.inputs[start:end]

        # Get u_out mask
        u_out = self.u_outs[start:end]

        if self.mode != "test":
            y = self.targets[start:end]
            return torch.tensor(x), torch.tensor(y), torch.tensor(u_out)
        else:
            # Return dummy targets for test set
            return torch.tensor(x), torch.zeros(self.seq_len), torch.tensor(u_out)


def add_features(df):
    """
    Generates physics-based features for the ventilator dataset.
    """
    # Ensure data is sorted to correctly calculate diffs and shifts
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # --- Time Integration ---
    # Calculate time delta (dt)
    # Group by breath_id to avoid negative diffs at breath boundaries
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    # Clip negative values just in case (though sorting should prevent this)
    df["dt"] = df["dt"].clip(lower=0)

    # Calculate Volume (Integral of u_in * dt)
    df["u_in_area"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in_area"].cumsum()

    # --- Physics Interactions ---
    # Resistive Pressure component: Flow * Resistance
    df["R_u_in"] = df["R"] * df["u_in"]

    # Elastic Pressure component: Volume / Compliance
    df["u_in_cumsum_C"] = df["u_in_cumsum"] / df["C"]

    # --- Dynamics (Lags and Diffs) ---
    # Using groupby to prevent data leakage between breaths
    for lag in [1, 2]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    for diff in [1, 2]:
        df[f"u_in_diff{diff}"] = df.groupby("breath_id")["u_in"].diff(diff).fillna(0)

    # Cleanup intermediate columns
    df = df.drop(columns=["u_in_area"])

    # Final fillna to ensure no NaNs remain
    df = df.fillna(0)

    return df


def get_data_loaders(load_cached_data=True):
    """
    Main function to load data, process features, and return DataLoaders.
    Handles caching to speed up subsequent runs.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_parquet = Config.TRAIN_CACHE
    val_parquet = Config.VAL_CACHE
    test_parquet = Config.TEST_CACHE
    scaler_path = Config.SCALER_PARAMS_PATH

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_parquet)
        and os.path.exists(val_parquet)
        and os.path.exists(test_parquet)
        and os.path.exists(scaler_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)
        test_df = pd.read_parquet(test_parquet)
    else:
        print("Cache not found or reload requested. Processing data from scratch...")

        # Load Raw Data
        print(f"Loading {Config.TRAIN_CSV}...")
        train_raw = pd.read_csv(Config.TRAIN_CSV)
        print(f"Loading {Config.TEST_CSV}...")
        test_raw = pd.read_csv(Config.TEST_CSV)

        # Load Metadata for Splitting
        print("Loading metadata...")
        train_meta = pd.read_csv(Config.TRAIN_META)
        val_meta = pd.read_csv(Config.VAL_META)

        train_breaths = set(train_meta["breath_id"].unique())
        val_breaths = set(val_meta["breath_id"].unique())

        # Split Train/Val
        print("Splitting train/val...")
        train_df = train_raw[train_raw["breath_id"].isin(train_breaths)].copy()
        val_df = train_raw[train_raw["breath_id"].isin(val_breaths)].copy()
        test_df = test_raw.copy()

        # Debug Mode: Subsample data
        if Config.DEBUG:
            print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths per split.")
            train_ids = list(train_breaths)[: Config.DEBUG_SAMPLE_SIZE]
            val_ids = list(val_breaths)[: Config.DEBUG_SAMPLE_SIZE]
            test_ids = test_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]

            train_df = train_df[train_df["breath_id"].isin(train_ids)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_ids)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_ids)].copy()

        # Feature Engineering
        print("Applying feature engineering...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Scaling
        print("Fitting RobustScaler...")
        # Identify continuous columns (exclude binary 'u_out')
        feature_cols = Config.INPUT_FEATURES
        continuous_cols = [c for c in feature_cols if c != "u_out"]

        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        scaler.fit(train_df[continuous_cols])

        # Transform all splits
        print("Transforming data...")
        train_df[continuous_cols] = scaler.transform(train_df[continuous_cols])
        val_df[continuous_cols] = scaler.transform(val_df[continuous_cols])
        test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

        # Save Scaler Params
        np.savez(scaler_path, center=scaler.center_, scale=scaler.scale_)

        # Save processed data to cache
        print("Saving data to cache...")
        train_df.to_parquet(train_parquet)
        val_df.to_parquet(val_parquet)
        test_df.to_parquet(test_parquet)

    # Create Datasets
    print("Creating Datasets...")
    train_dataset = VentilatorDataset(train_df, mode="train")
    val_dataset = VentilatorDataset(val_df, mode="val")
    test_dataset = VentilatorDataset(test_df, mode="test")

    # Create DataLoaders
    print("Creating DataLoaders...")
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

    return train_loader, val_loader, test_loader

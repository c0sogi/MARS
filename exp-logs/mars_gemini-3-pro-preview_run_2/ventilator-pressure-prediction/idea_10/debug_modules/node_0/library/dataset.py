import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements Time-Weighted Physics Engineering.
    Calculates volume via integration, physics interaction terms, and dynamics.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL]).reset_index(drop=True)

    # 1. Time-Weighted Integration (Volume)
    # Calculate dt (time difference between steps)
    # Group by breath_id to avoid diffing across breaths
    df["dt"] = df.groupby(Config.BREATH_ID_COL)[Config.TIME_COL].diff().fillna(0)

    # Volume = Integral(Flow * dt)
    df["u_in_area"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby(Config.BREATH_ID_COL)["u_in_area"].cumsum()

    # 2. Physics Interaction Terms
    # Resistive Pressure ~ R * Flow
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure ~ Volume / Compliance
    df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]

    # 3. Explicit Dynamics (Lags and Differences)
    # Lag 1 and 2
    df["u_in_lag1"] = df.groupby(Config.BREATH_ID_COL)["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = df.groupby(Config.BREATH_ID_COL)["u_in"].shift(2).fillna(0)

    # Finite Differences (Velocity and Acceleration)
    df["u_in_diff1"] = df.groupby(Config.BREATH_ID_COL)["u_in"].diff(1).fillna(0)
    df["u_in_diff2"] = df.groupby(Config.BREATH_ID_COL)["u_in"].diff(2).fillna(0)

    # Cleanup temporary columns
    df = df.drop(columns=["dt", "u_in_area"], errors="ignore")

    return df


class VentilatorDataset(Dataset):
    def __init__(self, X: np.ndarray, u_out: np.ndarray, y: np.ndarray = None):
        """
        Args:
            X: Input features of shape (Num_Breaths, Breath_Length, Num_Features)
            u_out: Binary control input of shape (Num_Breaths, Breath_Length)
            y: Target pressure of shape (Num_Breaths, Breath_Length). Optional (for test set).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx], "u_out": self.u_out[idx]}
        if self.y is not None:
            data["y"] = self.y[idx]
        return data


def load_and_preprocess_data(load_cached_data: bool = True):
    """
    Loads data, performs feature engineering, scales features, and returns Datasets.
    Implements caching using Parquet files.
    """
    seed_everything(Config.SEED)

    # Define cache paths based on Debug mode to avoid overwriting full cache with debug data
    suffix = "_debug.parquet" if Config.DEBUG else ".parquet"
    train_cache = Config.TRAIN_CACHE.replace(".parquet", suffix)
    val_cache = Config.VAL_CACHE.replace(".parquet", suffix)
    test_cache = Config.TEST_CACHE.replace(".parquet", suffix)

    # --- Helper to load or process a split ---
    def get_split_data(
        split_name, cache_path, metadata_path, raw_csv_path, is_test=False
    ):
        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # 2. Process from Scratch
            print(f"Processing {split_name} data from raw files...")

            # Load Metadata
            meta = pd.read_csv(metadata_path)
            if Config.DEBUG:
                # Sample breaths for debugging
                unique_breaths = meta[Config.BREATH_ID_COL].unique()
                sample_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
                meta = meta[meta[Config.BREATH_ID_COL].isin(sample_breaths)]

            # Load Raw Data
            # We load the full raw file and filter, or load selectively.
            # Given memory constraints and pandas efficiency, loading full then filtering is usually okay for 5GB.
            # However, to be safe, we can read the raw file and filter.
            raw_df = pd.read_csv(raw_csv_path)

            # Filter raw data to include only breaths in this split
            target_breaths = meta[Config.BREATH_ID_COL].unique()
            df = raw_df[raw_df[Config.BREATH_ID_COL].isin(target_breaths)].copy()

            # Engineer Features
            df = engineer_features(df)

            # Ensure columns are ordered and present
            # We keep ID and Breath ID for sorting/reshaping, and Target if present

            # Save to Cache
            print(f"Saving {split_name} data to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)

        return df

    # --- 1. Load/Process Dataframes ---
    train_df = get_split_data(
        "train", train_cache, Config.TRAIN_METADATA, Config.TRAIN_CSV
    )
    val_df = get_split_data("val", val_cache, Config.VAL_METADATA, Config.TRAIN_CSV)
    test_df = get_split_data(
        "test", test_cache, Config.TEST_METADATA, Config.TEST_CSV, is_test=True
    )

    # --- 2. Scaling ---
    print("Scaling features...")
    feature_cols = Config.FEATURE_COLS

    # Fit scaler on Train only
    scaler = RobustScaler()

    # We need to extract the feature values to fit
    # Note: We scale the columns in place or create new ones.
    # To save memory, we'll modify the dataframes.

    scaler.fit(train_df[feature_cols].values)

    train_df[feature_cols] = scaler.transform(train_df[feature_cols].values)
    val_df[feature_cols] = scaler.transform(val_df[feature_cols].values)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    # --- 3. Reshaping and Dataset Creation ---
    print("Reshaping data for Sequence Model...")

    def create_dataset(df, is_test=False):
        # Sort just in case
        df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL])

        # Calculate dimensions
        num_breaths = df[Config.BREATH_ID_COL].nunique()
        breath_len = 80  # Standard for this dataset

        # Extract arrays
        # Features
        X_flat = df[feature_cols].values
        X = X_flat.reshape(num_breaths, breath_len, len(feature_cols))

        # u_out (for loss weighting)
        u_out_flat = df["u_out"].values
        u_out = u_out_flat.reshape(num_breaths, breath_len)

        # Target
        y = None
        if not is_test and Config.TARGET_COL in df.columns:
            y_flat = df[Config.TARGET_COL].values
            y = y_flat.reshape(num_breaths, breath_len)

        return VentilatorDataset(X, u_out, y)

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df, is_test=True)

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

from library import config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Holds data in (N, 80, Features) format.
    """

    def __init__(self, X, u_out, y=None):
        self.X = torch.FloatTensor(X)
        self.u_out = torch.FloatTensor(u_out)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx], "u_out": self.u_out[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


def engineer_features(df):
    """
    Computes physical, lookahead, and positional encoding features.
    """
    # Ensure sorted by breath_id and id/time_step
    df = df.sort_values(by=[config.BREATH_ID_COL, config.ID_COL]).reset_index(drop=True)

    # 1. Physical Features
    # dt: Time delta
    df["dt"] = df.groupby(config.BREATH_ID_COL)[config.TIME_COL].diff().fillna(0)

    # area: Cumulative integration of u_in * dt
    # We calculate volume increment first, then cumsum
    df["volume_inc"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby(config.BREATH_ID_COL)["volume_inc"].cumsum()
    df.drop(columns=["volume_inc"], inplace=True)

    # du_in: Derivative of u_in
    df["du_in"] = df.groupby(config.BREATH_ID_COL)["u_in"].diff().fillna(0)

    # Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    df["area_C"] = df["area"] / df["C"]

    # 2. Lookahead Features
    # Shift u_in backwards to get future values (t+1 ... t+4)
    for i in range(1, config.LOOKAHEAD_STEPS + 1):
        col_name = f"u_in_next_{i}"
        df[col_name] = df.groupby(config.BREATH_ID_COL)["u_in"].shift(-i).fillna(0)

    return df


def prepare_data(load_cached_data=True, debug=False):
    """
    Main data processing pipeline.
    Loads data, engineers features, scales, reshapes, and returns DataLoaders.
    """
    seed_everything()

    # 1. Check Cache
    cache_exists = (
        os.path.exists(config.TRAIN_CACHE)
        and os.path.exists(config.VAL_CACHE)
        and os.path.exists(config.TEST_CACHE)
        and os.path.exists(config.SCALER_CACHE)
    )

    # Cite debug_lesson_4: Invalidate Data Caches When Modifying Feature Definitions
    # Check for stale debug cache when requesting full data
    if load_cached_data and cache_exists and not debug:
        # Peek at the cache to verify dataset size
        check_df = pd.read_parquet(config.TRAIN_CACHE, columns=[config.BREATH_ID_COL])
        # Full dataset has >50k breaths; debug has ~100. Threshold of 1000 is safe.
        if check_df[config.BREATH_ID_COL].nunique() < 1000:
            print(
                "Stale debug cache detected. Invalidating to regenerate full dataset..."
            )
            load_cached_data = False

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(config.TRAIN_CACHE)
        val_df = pd.read_parquet(config.VAL_CACHE)
        test_df = pd.read_parquet(config.TEST_CACHE)
        # Scaler is loaded only if needed for inverse transform, but here we just need data
    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH)
        test_df = pd.read_csv(config.TEST_PATH)

        if debug:
            print("Debug mode: Sampling data...")
            # Sample breaths, not random rows, to maintain sequence integrity
            train_breaths = train_df[config.BREATH_ID_COL].unique()
            val_breaths = val_df[config.BREATH_ID_COL].unique()

            # Take first 100 breaths for debug
            train_df = train_df[
                train_df[config.BREATH_ID_COL].isin(train_breaths[:100])
            ]
            val_df = val_df[val_df[config.BREATH_ID_COL].isin(val_breaths[:50])]
            # Test set usually doesn't need sampling for debug unless very slow, but let's keep it full or sample
            test_breaths = test_df[config.BREATH_ID_COL].unique()
            test_df = test_df[test_df[config.BREATH_ID_COL].isin(test_breaths[:50])]

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Scaling
        print("Fitting Scaler...")
        scaler = RobustScaler()

        # Exclude u_out from scaling to preserve binary 0/1 semantics
        scale_cols = [c for c in config.FEATURE_COLS if c != "u_out"]

        # Fit only on training data
        scaler.fit(train_df[scale_cols])

        # Transform all sets
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Save to Cache
        print("Saving to cache...")
        train_df.to_parquet(config.TRAIN_CACHE)
        val_df.to_parquet(config.VAL_CACHE)
        test_df.to_parquet(config.TEST_CACHE)
        joblib.dump(scaler, config.SCALER_CACHE)

    # 2. Reshape to (N_breaths, 80, N_features)
    # We assume each breath has exactly 80 time steps.
    # The dataset is usually structured this way.
    SEQ_LEN = 80

    def reshape_data(df, is_test=False):
        # Ensure data is sorted
        df = df.sort_values(by=[config.BREATH_ID_COL, config.ID_COL])

        num_breaths = len(df) // SEQ_LEN
        if len(df) % SEQ_LEN != 0:
            raise ValueError(
                f"Data length {len(df)} is not divisible by sequence length {SEQ_LEN}"
            )

        feature_vals = df[config.FEATURE_COLS].values.astype(np.float32)
        u_out_vals = df["u_out"].values.astype(np.float32)

        X = feature_vals.reshape(num_breaths, SEQ_LEN, -1)
        u_out = u_out_vals.reshape(num_breaths, SEQ_LEN)

        y = None
        if not is_test:
            y_vals = df[config.TARGET_COL].values.astype(np.float32)
            y = y_vals.reshape(num_breaths, SEQ_LEN)

        return X, u_out, y

    print("Reshaping data for Sequence Models...")
    X_train, u_out_train, y_train = reshape_data(train_df, is_test=False)
    X_val, u_out_val, y_val = reshape_data(val_df, is_test=False)
    X_test, u_out_test, _ = reshape_data(test_df, is_test=True)

    # 3. Create Datasets and Loaders
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    test_dataset = VentilatorDataset(X_test, u_out_test, None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    return train_loader, val_loader, test_loader

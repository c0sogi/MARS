import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config


class SegregatedScaler:
    """
    Applies RobustScaler to continuous features while keeping binary features raw.
    Avoids the 'Silent Peril' of scaling binary flags.
    """

    def __init__(self):
        self.continuous_features = Config.CONTINUOUS_FEATURES
        self.binary_features = Config.BINARY_FEATURES
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, df):
        if not self.continuous_features:
            return
        self.scaler.fit(df[self.continuous_features])
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")

        # Scale continuous features
        X_cont = self.scaler.transform(df[self.continuous_features])

        # Get binary features raw
        X_bin = df[self.binary_features].values

        # Concatenate: Continuous first, then Binary
        return np.concatenate([X_cont, X_bin], axis=1).astype(np.float32)

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        np.savez(path, center=self.scaler.center_, scale=self.scaler.scale_)

    def load(self, path):
        data = np.load(path)
        self.scaler.center_ = data["center"]
        self.scaler.scale_ = data["scale"]
        self.is_fitted = True


def feature_engineering(df):
    """
    Generates explicit physics terms and dynamic features.
    """
    # Ensure sorted order for sequential calculations
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # 1. Time Delta (dt)
    # We can safely use diff because we group by breath_id or handle boundaries.
    # Groupby is safer.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume Integration (Area)
    # Volume = cumsum(u_in * dt)
    # Calculate term first
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby("breath_id")["u_in_dt"].cumsum()

    # 3. Physics Interactions
    # Resistive Pressure Proxy: R * u_in
    df["R__u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure Proxy: Volume / C
    df["u_in_cumsum_div_C"] = df["area"] / df["C"]

    # 4. Multi-step Deltas (Lags)
    # Lag 1 to 4
    grp = df.groupby("breath_id")["u_in"]
    for i in range(1, 5):
        df[f"u_in_lag{i}"] = grp.shift(i).fillna(0)

    # Cleanup intermediate columns if necessary, but RobustScaler will select by name
    return df


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (seq_len, num_features)
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.y is not None:
            # y shape: (seq_len,)
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor
        else:
            return x_tensor


def prepare_datasets(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and reshaping.
    Uses caching to speed up subsequent runs.
    """
    Config.setup()

    cache_dir = Config.WORKING_DIR
    train_cache = os.path.join(cache_dir, "train_data.npz")
    val_cache = os.path.join(cache_dir, "val_data.npz")
    test_cache = os.path.join(cache_dir, "test_data.npz")
    scaler_cache = os.path.join(cache_dir, "scaler_params.npz")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(scaler_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading cached datasets...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)

        X_train, y_train = train_data["X"], train_data["y"]
        X_val, y_val = val_data["X"], val_data["y"]
        X_test = test_data["X"]  # No y for test

        # Load scaler just to ensure state consistency if needed later, though data is already scaled
        scaler = SegregatedScaler()
        scaler.load(scaler_cache)

    else:
        print("Processing data from scratch...")

        # 1. Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_META)
        val_meta = pd.read_csv(Config.VAL_META)
        test_meta = pd.read_csv(Config.TEST_META)

        train_breath_ids = set(train_meta["breath_id"].unique())
        val_breath_ids = set(val_meta["breath_id"].unique())

        # 2. Load Raw Data
        # Optimization: Load full train and split in memory
        df_train_full = pd.read_csv(Config.TRAIN_CSV)
        df_test = pd.read_csv(Config.TEST_CSV)

        # Split Train/Val
        df_train = df_train_full[
            df_train_full["breath_id"].isin(train_breath_ids)
        ].copy()
        df_val = df_train_full[df_train_full["breath_id"].isin(val_breath_ids)].copy()

        del df_train_full  # Free memory

        # 3. Feature Engineering
        print("Applying feature engineering...")
        df_train = feature_engineering(df_train)
        df_val = feature_engineering(df_val)
        df_test = feature_engineering(df_test)

        # 4. Scaling
        print("Fitting and applying SegregatedScaler...")
        scaler = SegregatedScaler()
        scaler.fit(df_train)

        X_train_flat = scaler.transform(df_train)
        X_val_flat = scaler.transform(df_val)
        X_test_flat = scaler.transform(df_test)

        y_train_flat = df_train[Config.TARGET_COL].values
        y_val_flat = df_val[Config.TARGET_COL].values
        # Test has no target

        # 5. Reshaping to Sequences
        # Assumption: Data is sorted by breath_id, time_step and each breath is 80 steps.
        # The metadata generation script ensures grouping, and feature_engineering sorts.
        SEQ_LEN = 80

        # Helper to reshape
        def reshape_seq(X_flat):
            # X_flat shape: (N_rows, n_feats) -> (N_breaths, 80, n_feats)
            if len(X_flat) % SEQ_LEN != 0:
                raise ValueError(f"Total rows {len(X_flat)} not divisible by {SEQ_LEN}")
            return X_flat.reshape(-1, SEQ_LEN, X_flat.shape[1])

        def reshape_target(y_flat):
            return y_flat.reshape(-1, SEQ_LEN)

        X_train = reshape_seq(X_train_flat)
        y_train = reshape_target(y_train_flat)

        X_val = reshape_seq(X_val_flat)
        y_val = reshape_target(y_val_flat)

        X_test = reshape_seq(X_test_flat)

        # 6. Save to Cache
        print("Saving datasets to cache...")
        np.savez(train_cache, X=X_train, y=y_train)
        np.savez(val_cache, X=X_val, y=y_val)
        np.savez(test_cache, X=X_test)
        scaler.save(scaler_cache)

    # Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test)  # No target

    print(f"Datasets prepared.")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    return train_dataset, val_dataset, test_dataset

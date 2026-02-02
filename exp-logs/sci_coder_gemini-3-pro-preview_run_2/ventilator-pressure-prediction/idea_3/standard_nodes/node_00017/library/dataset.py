import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.utils import seed_everything

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_3"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TRAIN_META = "train_metadata.csv"
VAL_META = "val_metadata.csv"
TEST_META = "test_metadata.csv"


class VentilatorDataset(Dataset):
    def __init__(self, X, u_out, y=None):
        """
        Args:
            X: Input features of shape (N, 80, F)
            u_out: Expiratory valve control of shape (N, 80) - used for metrics
            y: Target pressure of shape (N, 80)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"input": self.X[idx], "u_out": self.u_out[idx]}
        if self.y is not None:
            data["target"] = self.y[idx]
        return data


def feature_engineering(df):
    """
    Generates physics-based and dynamic features.
    """
    # Ensure sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # --- Physics Features ---
    # Cumulative volume approximation (integral of flow u_in)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # Interaction terms based on Equation of Motion
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # --- Dynamics (Lags and Diffs) ---
    # Using groupby shift for safety across breath boundaries
    grp = df.groupby("breath_id")["u_in"]

    df["u_in_lag1"] = grp.shift(1).fillna(0)
    df["u_in_lag2"] = grp.shift(2).fillna(0)
    df["u_in_lag_back1"] = grp.shift(-1).fillna(0)
    df["u_in_lag_back2"] = grp.shift(-2).fillna(0)

    # Finite differences (Approximation of derivatives)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - (df["u_in_lag1"] - df["u_in_lag2"])

    # --- Categorical Mappings ---
    # Map R and C values to 0-based indices for embedding layers
    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    df["R_cat"] = df["R"].map(r_map)
    df["C_cat"] = df["C"].map(c_map)

    return df


def preprocess_data(load_cached_data=True, debug=False):
    """
    Loads, processes, scales, and reshapes data. Handles caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(CACHE_DIR, "test_data.npz")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading cached data from {CACHE_DIR}...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)
        return (
            train_data["X"],
            train_data["y"],
            train_data["u_out"],
            val_data["X"],
            val_data["y"],
            val_data["u_out"],
            test_data["X"],
            test_data["u_out"],
        )

    print("Cache not found or reload forced. Processing from scratch...")

    # 2. Load Metadata and Raw Data
    train_meta_df = pd.read_csv(os.path.join(METADATA_DIR, TRAIN_META))
    val_meta_df = pd.read_csv(os.path.join(METADATA_DIR, VAL_META))

    train_breath_ids = set(train_meta_df["breath_id"].unique())
    val_breath_ids = set(val_meta_df["breath_id"].unique())

    print("Loading raw CSV files...")
    df_train_full = pd.read_csv(os.path.join(INPUT_DIR, TRAIN_FILE))
    df_test = pd.read_csv(os.path.join(INPUT_DIR, TEST_FILE))

    # Debug Mode: Subsample
    if debug:
        print("Debug mode: Subsampling data...")
        train_breath_ids = set(list(train_breath_ids)[:100])
        val_breath_ids = set(list(val_breath_ids)[:20])
        df_train_full = df_train_full[
            df_train_full["breath_id"].isin(train_breath_ids | val_breath_ids)
        ]
        test_ids = df_test["breath_id"].unique()[:50]
        df_test = df_test[df_test["breath_id"].isin(test_ids)]

    # 3. Feature Engineering
    print("Applying feature engineering...")
    df_train_full = feature_engineering(df_train_full)
    df_test = feature_engineering(df_test)

    # 4. Split Train/Val
    df_train = df_train_full[df_train_full["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_train_full[df_train_full["breath_id"].isin(val_breath_ids)].copy()
    del df_train_full

    # 5. Scaling
    # Define continuous columns to scale. Exclude u_out, R_cat, C_cat.
    scale_cols = [
        "time_step",
        "u_in",
        "u_in_cumsum",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag_back1",
        "u_in_lag_back2",
        "u_in_diff1",
        "u_in_diff2",
        "R_u_in",
        "vol_C",
    ]

    print("Scaling continuous features...")
    scaler = RobustScaler()
    df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
    df_val[scale_cols] = scaler.transform(df_val[scale_cols])
    df_test[scale_cols] = scaler.transform(df_test[scale_cols])

    # 6. Reshape to (N, 80, F)
    # Final feature list: Scaled continuous + u_out + Categorical Indices
    # Note: u_out is kept unscaled (0/1).
    final_cols = scale_cols + ["u_out", "R_cat", "C_cat"]

    def reshape_dataset(df, target_col=None):
        df = df.sort_values(["breath_id", "time_step"])
        n_breaths = len(df) // 80

        # Features
        X = df[final_cols].values.reshape(n_breaths, 80, len(final_cols))

        # u_out (for metric masking)
        u_out = df["u_out"].values.reshape(n_breaths, 80)

        if target_col:
            y = df[target_col].values.reshape(n_breaths, 80)
            return X, y, u_out
        else:
            return X, u_out

    print("Reshaping tensors...")
    X_train, y_train, u_out_train = reshape_dataset(df_train, "pressure")
    X_val, y_val, u_out_val = reshape_dataset(df_val, "pressure")
    X_test, u_out_test = reshape_dataset(df_test, None)

    # 7. Save to Cache
    print(f"Saving processed data to {CACHE_DIR}...")
    np.savez(train_cache, X=X_train, y=y_train, u_out=u_out_train)
    np.savez(val_cache, X=X_val, y=y_val, u_out=u_out_val)
    np.savez(test_cache, X=X_test, u_out=u_out_test)

    return X_train, y_train, u_out_train, X_val, y_val, u_out_val, X_test, u_out_test


def get_data_loaders(batch_size, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    data = preprocess_data(load_cached_data=load_cached_data, debug=debug)
    X_train, y_train, u_out_train, X_val, y_val, u_out_val, X_test, u_out_test = data

    train_ds = VentilatorDataset(X_train, u_out_train, y_train)
    val_ds = VentilatorDataset(X_val, u_out_val, y_val)
    test_ds = VentilatorDataset(X_test, u_out_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader

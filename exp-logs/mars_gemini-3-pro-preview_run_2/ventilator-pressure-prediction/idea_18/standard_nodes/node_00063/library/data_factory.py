import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import set_seed


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.u_out[idx]
        return self.X[idx]


def add_features(df):
    """
    Computes physics-inspired features and dynamics based on Idea 18.
    """
    # 1. Time-Weighted Integration
    # Calculate dt (time difference between steps). Fill first step with 0.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0.0)

    # Calculate Volume: Integral of Flow * dt
    # u_in is proportional to flow.
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["volume"] = df.groupby("breath_id")["u_in_dt"].cumsum()

    # 2. Physics Interaction Terms
    df["R_u_in"] = df["R"] * df["u_in"]  # Resistive component proxy
    df["vol_C"] = df["volume"] / df["C"]  # Elastic component proxy
    df["R_div_C"] = df["R"] / df["C"]  # Time constant proxy

    # 3. Explicit Dynamics (Past Lags & Finite Differences)
    # Exclude future lags to prevent leakage.
    for lag in [1, 2, 3, 4]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Finite differences
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    return df


def prepare_data(load_cached_data=True):
    """
    Loads data, generates features, applies segregated scaling, and reshapes for RNN.
    Implements caching mechanism in ./working/idea_18/
    """
    set_seed(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(Config.CACHE_DIR, "train_processed.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_processed.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_processed.npz")
    scaler_cache = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    # --- 1. Cache Loading ---
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading cached data from", Config.CACHE_DIR)
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)
        return (
            (train_data["X"], train_data["y"], train_data["u_out"]),
            (val_data["X"], val_data["y"], val_data["u_out"]),
            (test_data["X"], test_data["ids"]),
        )

    print("Processing data from scratch...")

    # --- 2. Data Loading & Splitting ---
    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)

    # Load Raw Data
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Identify Breath IDs for Split
    train_breath_ids = train_meta["breath_id"].unique()
    val_breath_ids = val_meta["breath_id"].unique()

    # Split Train/Val
    df_train = df_train_full[df_train_full["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_train_full[df_train_full["breath_id"].isin(val_breath_ids)].copy()

    # Cleanup
    del df_train_full
    gc.collect()

    # --- 3. Feature Engineering ---
    print("Generating features...")
    df_train = add_features(df_train)
    df_val = add_features(df_val)
    df_test = add_features(df_test)

    # --- 4. Segregated Scaling ---
    print("Applying segregated scaling...")

    # Define columns
    exclude_cols = ["id", "breath_id", "pressure", "u_out", "u_in_dt", "dt"]
    continuous_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Continuous features to scale: {continuous_cols}")

    # Initialize Scaler
    scaler = RobustScaler()

    # Fit on Train Continuous only
    X_train_cont = scaler.fit_transform(df_train[continuous_cols].values)
    X_val_cont = scaler.transform(df_val[continuous_cols].values)
    X_test_cont = scaler.transform(df_test[continuous_cols].values)

    # Extract Binary u_out (Raw)
    u_out_train = df_train["u_out"].values.reshape(-1, 1)
    u_out_val = df_val["u_out"].values.reshape(-1, 1)
    u_out_test = df_test["u_out"].values.reshape(-1, 1)

    # Concatenate: [Scaled Continuous, Raw Binary]
    # u_out is the last feature
    X_train = np.hstack([X_train_cont, u_out_train])
    X_val = np.hstack([X_val_cont, u_out_val])
    X_test = np.hstack([X_test_cont, u_out_test])

    # Extract Targets
    y_train = df_train["pressure"].values
    y_val = df_val["pressure"].values

    # Extract IDs for Test (flat)
    test_ids = df_test["id"].values

    # --- 5. Reshaping for LSTM ---
    # Shape: (N_breaths, 80, N_features)
    # We assume data is sorted by breath_id and time_step (default structure)

    def reshape_to_seq(X, y=None):
        num_breaths = X.shape[0] // 80
        X_seq = X.reshape(num_breaths, 80, -1)
        if y is not None:
            y_seq = y.reshape(num_breaths, 80)
            return X_seq, y_seq
        return X_seq

    print("Reshaping tensors...")
    X_train_seq, y_train_seq = reshape_to_seq(X_train, y_train)
    X_val_seq, y_val_seq = reshape_to_seq(X_val, y_val)
    X_test_seq = reshape_to_seq(X_test)

    # Extract u_out sequences for loss weighting (N, 80)
    # u_out is the last column in X
    u_out_train_seq = X_train_seq[:, :, -1]
    u_out_val_seq = X_val_seq[:, :, -1]

    # --- 6. Caching ---
    print("Saving to cache...")
    np.savez(train_cache, X=X_train_seq, y=y_train_seq, u_out=u_out_train_seq)
    np.savez(val_cache, X=X_val_seq, y=y_val_seq, u_out=u_out_val_seq)
    np.savez(test_cache, X=X_test_seq, ids=test_ids)

    # Save scaler params
    np.savez(scaler_cache, center=scaler.center_, scale=scaler.scale_)

    return (
        (X_train_seq, y_train_seq, u_out_train_seq),
        (X_val_seq, y_val_seq, u_out_val_seq),
        (X_test_seq, test_ids),
    )

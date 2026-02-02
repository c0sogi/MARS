import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None):
        """
        PyTorch Dataset for Ventilator Pressure Prediction.
        X: Feature tensor of shape (N_breaths, Sequence_Length, N_features)
        y: Target tensor of shape (N_breaths, Sequence_Length)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def compute_features(df):
    """
    Applies Time-Weighted Physics Engineering to the dataframe.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # 1. Time Delta (dt)
    # Calculate difference in time_step, filling the first step of each breath with 0
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume Integration (Time-Weighted)
    # Volume is the integral of flow (u_in) over time.
    # u_in ranges 0-100.
    if Config.USE_TIME_WEIGHTED_INTEGRATION:
        # Vectorized cumulative sum by group
        df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    else:
        df["volume"] = df.groupby("breath_id")["u_in"].cumsum()

    # 3. Physics Interaction Terms
    if Config.USE_PHYSICS_INTERACTIONS:
        # Resistive Pressure component ~ R * Flow
        df["R_u_in"] = df["R"] * df["u_in"]
        # Elastic Pressure component ~ Volume / Compliance
        df["vol_C"] = df["volume"] / df["C"]
        # Interactions between lung attributes
        df["R_div_C"] = df["R"] / df["C"]
        df["R_times_C"] = df["R"] * df["C"]

    # 4. Dynamics (Lags and Derivatives)
    if Config.USE_LAG_FEATURES:
        for lag in range(1, 5):
            df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    if Config.USE_DERIVATIVES:
        # 1st Derivative of Flow (Acceleration)
        df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
        # 2nd Derivative of Flow (Jerk)
        df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    # 5. Additional Features
    # Cumulative time can be useful for positional encoding implicitly
    df["time_cumsum"] = df.groupby("breath_id")["dt"].cumsum()

    # u_out is already in the dataframe, we ensure it's kept as a feature

    return df


def get_scaler_stats(X):
    """
    Computes Mean and Std for StandardScaler logic.
    X shape: (N, Seq, Feat)
    """
    # Flatten N and Seq dimensions for stats calculation
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    # Prevent division by zero for constant features (like u_out might be in short segments, though unlikely globally)
    std[std == 0] = 1.0
    return mean, std


def apply_scaling(X, mean, std):
    return (X - mean) / std


def load_and_process_data(split, load_cached_data=True):
    """
    Loads raw data, processes features, scales, reshapes, and caches the result.
    """
    # Define cache paths
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_processed.npz")
    scaler_file = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        data = np.load(cache_file)
        X = data["X"]
        y = data["y"] if "y" in data.files else None
        ids = data["ids"]
        return X, y, ids

    print(f"Processing {split} data from scratch...")

    # 2. Determine Source Files
    if split == "train":
        meta_path = Config.TRAIN_META
        source_csv = Config.TRAIN_CSV
    elif split == "val":
        meta_path = Config.VAL_META
        source_csv = Config.TRAIN_CSV
    else:  # test
        meta_path = Config.TEST_META
        source_csv = Config.TEST_CSV

    # 3. Load Data
    # Load metadata to get the specific breath_ids for this split
    df_meta = pd.read_csv(meta_path)
    target_breath_ids = df_meta["breath_id"].unique()

    # Load raw source data
    # We load the full file. 5M rows fits in memory.
    df_source = pd.read_csv(source_csv)

    # Filter for the specific split
    df = df_source[df_source["breath_id"].isin(target_breath_ids)].copy()

    # Sort to ensure deterministic order for reshaping
    df = df.sort_values(["breath_id", "id"])

    # 4. Feature Engineering
    df = compute_features(df)

    # 5. Prepare Tensors
    # Identify feature columns (exclude meta/target)
    # We keep u_out as a feature
    exclude_cols = ["id", "breath_id", "pressure"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = sorted(feature_cols)  # Deterministic order

    # Reshape to (N_breaths, 80, N_features)
    # Assumption: All breaths have 80 time steps (standard for this dataset)
    n_breaths = len(target_breath_ids)
    seq_len = 80

    # Safety check
    if len(df) != n_breaths * seq_len:
        print(
            f"Warning: Data length {len(df)} is not exactly {n_breaths} * {seq_len}. Reshaping might fail or be misaligned."
        )

    X_raw = df[feature_cols].values.reshape(n_breaths, seq_len, -1)

    y = None
    if "pressure" in df.columns:
        y = df["pressure"].values.reshape(n_breaths, seq_len)

    ids = df["id"].values  # Flattened IDs corresponding to the sequence

    # 6. Scaling
    # We fit scaler ONLY on train data, apply to val/test
    if split == "train":
        mean, std = get_scaler_stats(X_raw)
        np.savez(scaler_file, mean=mean, std=std)
    else:
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"Scaler parameters not found at {scaler_file}. Please process 'train' split first."
            )
        scaler_data = np.load(scaler_file)
        mean = scaler_data["mean"]
        std = scaler_data["std"]

    X_scaled = apply_scaling(X_raw, mean, std)

    # 7. Save to Cache
    save_dict = {"X": X_scaled, "ids": ids}
    if y is not None:
        save_dict["y"] = y

    np.savez(cache_file, **save_dict)

    return X_scaled, y, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Orchestrates data loading and returns PyTorch DataLoaders.
    Also updates Config.INPUT_DIM based on processed features.
    """
    # 1. Process Train (Must be first to generate scaler)
    X_train, y_train, _ = load_and_process_data("train", load_cached_data)

    # Update Config with dynamic input dimension
    Config.INPUT_DIM = X_train.shape[2]

    # 2. Process Val
    X_val, y_val, _ = load_and_process_data("val", load_cached_data)

    # 3. Process Test
    X_test, _, test_ids = load_and_process_data("test", load_cached_data)

    # 4. Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader, test_ids

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


def add_features(df):
    """
    Adds physics-informed features and time-series derivatives to the dataframe.
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Group by breath_id for window operations
    # Using transform is generally faster than apply for simple operations
    grp = df.groupby("breath_id")

    # 1. Physics-Based Integration (Air Volume)
    df["u_in_cumsum"] = grp["u_in"].cumsum()

    # 2. Equation of Motion Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # 3. Explicit Dynamics (Lags)
    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0)

    # 4. Explicit Dynamics (Finite Differences)
    df["u_in_diff1"] = grp["u_in"].diff(1).fillna(0)
    df["u_in_diff2"] = grp["u_in"].diff(2).fillna(0)

    # 5. Time delta
    df["dt"] = grp["time_step"].diff(1).fillna(0)

    return df


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        # u_out is needed for weighted loss calculation
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx]}
        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]
        if self.y is not None:
            data["y"] = self.y[idx]
        return data


def prepare_data(split, config, load_cached_data=True, debug=False, limit_breaths=None):
    """
    Orchestrates data loading, processing, scaling, and DataLoader creation.

    Args:
        split (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to load from disk cache if available.
        debug (bool): If True, appends '_debug' to cache filenames.
        limit_breaths (int, optional): Number of breaths to load for debugging/testing.

    Returns:
        DataLoader: PyTorch DataLoader for the requested split.
    """
    # Determine cache paths
    suffix = "_debug" if debug else ""
    cache_file = os.path.join(config.WORKING_DIR, f"{split}_data{suffix}.npz")
    scaler_file = os.path.join(config.WORKING_DIR, f"scaler_params{suffix}.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        data = np.load(cache_file)
        X, u_out = data["X"], data["u_out"]
        y = data["y"] if "y" in data else None

        dataset = VentilatorDataset(X, y, u_out)
        shuffle = split == "train"
        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    print(f"Processing {split} data from scratch...")

    # 2. Load Metadata and Raw Data
    if split == "train":
        meta_path = config.TRAIN_META
        data_path = config.TRAIN_PATH
    elif split == "val":
        meta_path = config.VAL_META
        data_path = config.TRAIN_PATH
    else:  # test
        meta_path = config.TEST_META
        data_path = config.TEST_PATH

    meta_df = pd.read_csv(meta_path)

    # Debugging: Limit number of breaths
    if limit_breaths is not None:
        unique_breaths = meta_df["breath_id"].unique()[:limit_breaths]
        meta_df = meta_df[meta_df["breath_id"].isin(unique_breaths)]

    # Load source data and filter by breath_ids in metadata
    # Optimizing memory: read only necessary rows if possible, but here we filter after read
    # for simplicity given dataset size fits in memory.
    source_df = pd.read_csv(data_path)
    target_breaths = meta_df["breath_id"].unique()
    df = source_df[source_df["breath_id"].isin(target_breaths)].copy()

    # 3. Feature Engineering
    df = add_features(df)

    # Define columns for scaling (exclude u_out, id, breath_id, pressure)
    feature_cols = [
        "time_step",
        "u_in",
        "u_in_cumsum",
        "R",
        "C",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "dt",
    ]

    # 4. Scaling
    data_matrix = df[feature_cols].values.astype(np.float32)

    if split == "train":
        scaler = RobustScaler()
        data_matrix = scaler.fit_transform(data_matrix)
        # Save scaler parameters
        np.savez(scaler_file, center=scaler.center_, scale=scaler.scale_)
    else:
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"Scaler parameters not found at {scaler_file}. Run train split first."
            )
        params = np.load(scaler_file)
        center = params["center"]
        scale = params["scale"]
        data_matrix = (data_matrix - center) / scale

    # 5. Reshaping and Formatting
    # Append u_out to features (not scaled)
    u_out_col = df["u_out"].values.reshape(-1, 1).astype(np.float32)
    X_flat = np.hstack([data_matrix, u_out_col])

    # Reshape to (Num_Breaths, 80, Num_Features)
    # Each breath is guaranteed to have 80 time steps in this dataset
    num_breaths = len(df) // 80
    num_features = X_flat.shape[1]

    X = X_flat.reshape(num_breaths, 80, num_features)
    u_out = df["u_out"].values.reshape(num_breaths, 80).astype(np.float32)

    if split != "test":
        y = df["pressure"].values.reshape(num_breaths, 80).astype(np.float32)
    else:
        y = None

    # 6. Save to Cache
    save_dict = {"X": X, "u_out": u_out}
    if y is not None:
        save_dict["y"] = y
    np.savez_compressed(cache_file, **save_dict)

    # 7. Create DataLoader
    dataset = VentilatorDataset(X, y, u_out)
    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

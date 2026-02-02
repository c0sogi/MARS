import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns:
        x (torch.Tensor): Features [Sequence_Length, Num_Features]
        u_out (torch.Tensor): Control input mask [Sequence_Length]
        y (torch.Tensor): Target pressure [Sequence_Length] (or zeros for test)
    """

    def __init__(self, x, u_out, y=None):
        self.x = torch.FloatTensor(x)
        self.u_out = torch.FloatTensor(u_out)
        self.y = torch.FloatTensor(y) if y is not None else torch.zeros_like(self.u_out)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.u_out[idx], self.y[idx]


def add_features(df):
    """
    Generates physical and lag features for the ventilator dataset.
    """
    # Ensure sorted by breath_id and time_step (though usually already sorted)
    # df = df.sort_values(['breath_id', 'time_step']) # Assuming input is already sorted for performance

    # 1. Time Delta (dt)
    # Vectorized calculation: diff of time_step, masked where breath_id changes
    df["dt"] = df["time_step"].diff()
    # Mask out the first element of each breath (where breath_id changes)
    # We assume the first row is start of a breath, and subsequent rows check change
    breath_change_mask = df["breath_id"] != df["breath_id"].shift(1)
    df.loc[breath_change_mask, "dt"] = 0.0
    df["dt"] = df["dt"].fillna(0.0)

    # 2. Volume (Area) Integration
    # area = cumsum(u_in * dt)
    # Groupby cumsum is reasonably fast
    if Config.USE_AREA:
        df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # 3. Acceleration (du_in)
    if Config.USE_DELTA_U_IN:
        df["du_in"] = df["u_in"].diff()
        df.loc[breath_change_mask, "du_in"] = 0.0
        df["du_in"] = df["du_in"].fillna(0.0)

    # 4. Lookahead Features (u_in next steps)
    for i in range(1, Config.LOOKAHEAD_STEPS + 1):
        col_name = f"u_in_next{i}"
        # Shift upwards
        df[col_name] = df["u_in"].shift(-i)
        # Mask where breath_id changes in the future
        # If we shift -1, we compare breath_id[t] with breath_id[t+1]
        future_mask = df["breath_id"] != df["breath_id"].shift(-i)
        df.loc[future_mask, col_name] = 0.0
        df[col_name] = df[col_name].fillna(0.0)

    # 5. Interaction Terms
    if Config.USE_INTERACTION:
        df["R_u_in"] = df["R"] * df["u_in"]
        # Avoid division by zero if C is 0 (unlikely in this dataset, C is 10, 20, 50)
        df["area_C"] = df["area"] / df["C"]

    return df


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.
    Returns:
        data_dict (dict): Contains 'train', 'val', 'test' dictionaries with keys 'x', 'y', 'u_out'.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE)
        and os.path.exists(Config.VAL_CACHE)
        and os.path.exists(Config.TEST_CACHE)
        and os.path.exists(Config.SCALER_PATH)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_df = pd.read_parquet(Config.TRAIN_CACHE)
        val_df = pd.read_parquet(Config.VAL_CACHE)
        test_df = pd.read_parquet(Config.TEST_CACHE)
        # Scaler is loaded just to ensure it exists, though we assume data in parquet is already scaled/processed
        # Actually, standard practice: save processed features.
        # We will assume the parquet files contain the FINAL processed features.
    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_raw = pd.read_csv(Config.TRAIN_CSV)
        val_raw = pd.read_csv(Config.VAL_CSV)
        test_raw = pd.read_csv(Config.TEST_CSV)

        # Feature Engineering
        print("Generating features...")
        train_df = add_features(train_raw)
        val_df = add_features(val_raw)
        test_df = add_features(test_raw)

        # Define columns to scale
        # We exclude u_out (binary), id, breath_id, pressure
        # We include physics features
        scale_cols = ["u_in", "R", "C", "dt"]
        if Config.USE_AREA:
            scale_cols.append("area")
        if Config.USE_DELTA_U_IN:
            scale_cols.append("du_in")
        if Config.USE_INTERACTION:
            scale_cols.extend(["R_u_in", "area_C"])
        for i in range(1, Config.LOOKAHEAD_STEPS + 1):
            scale_cols.append(f"u_in_next{i}")

        if not Config.EXCLUDE_RAW_TIME:
            scale_cols.append("time_step")

        # Scaling
        print("Fitting Scaler...")
        scaler = RobustScaler()
        train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Save Scaler
        joblib.dump(scaler, Config.SCALER_PATH)

        # Save to Cache (Parquet)
        print("Saving to cache...")
        train_df.to_parquet(Config.TRAIN_CACHE)
        val_df.to_parquet(Config.VAL_CACHE)
        test_df.to_parquet(Config.TEST_CACHE)

    # Reshaping for Sequence Models
    # Data needs to be (N_breaths, 80, N_features)
    # We assume 80 steps per breath. We can verify or enforce.
    # The dataset is strictly structured as 80 steps per breath.
    SEQ_LEN = 80

    def reshape_dataset(df, is_test=False):
        # Select Feature Columns
        # Exclude metadata and target
        # Cite solution_lesson_node_00072: Include u_out in features for bidirectional context
        exclude_cols = ["id", "breath_id", "pressure"]
        if Config.EXCLUDE_RAW_TIME:
            exclude_cols.append("time_step")

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Extract arrays
        x_flat = df[feature_cols].values.astype(np.float32)
        u_out_flat = df["u_out"].values.astype(np.float32)

        num_samples = len(df)
        num_breaths = num_samples // SEQ_LEN

        # Reshape
        x = x_flat.reshape(num_breaths, SEQ_LEN, -1)
        u_out = u_out_flat.reshape(num_breaths, SEQ_LEN)

        if not is_test:
            y_flat = df["pressure"].values.astype(np.float32)
            y = y_flat.reshape(num_breaths, SEQ_LEN)
        else:
            y = None

        return x, u_out, y

    print("Reshaping datasets...")
    train_x, train_u_out, train_y = reshape_dataset(train_df)
    val_x, val_u_out, val_y = reshape_dataset(val_df)
    test_x, test_u_out, _ = reshape_dataset(test_df, is_test=True)

    return {
        "train": {"x": train_x, "u_out": train_u_out, "y": train_y},
        "val": {"x": val_x, "u_out": val_u_out, "y": val_y},
        "test": {"x": test_x, "u_out": test_u_out, "y": None},
    }


def get_data_loaders(load_cached_data=True):
    """
    Factory function to create DataLoaders.
    """
    data = prepare_data(load_cached_data=load_cached_data)

    train_ds = VentilatorDataset(
        data["train"]["x"], data["train"]["u_out"], data["train"]["y"]
    )
    val_ds = VentilatorDataset(data["val"]["x"], data["val"]["u_out"], data["val"]["y"])
    test_ds = VentilatorDataset(
        data["test"]["x"], data["test"]["u_out"], data["test"]["y"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, test_loader

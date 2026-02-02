import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Holds data in memory as numpy arrays and converts to tensors on-the-fly.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to float32 tensors
        x_item = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y_item = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_item, y_item
        return x_item


def add_features(df):
    """
    Implements the Complete Kinematic State Engineering pipeline.
    Calculates derivatives, integrals, and lookahead features.
    """
    # Ensure data is sorted by breath_id and time_step (should be already, but safety first)
    # df = df.sort_values(['breath_id', 'time_step']) # Assuming input is already sorted for performance

    # 1. Time Delta (dt)
    # Group by breath_id to avoid diffing across breaths
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Backward Velocity (Momentum): u_in(t) - u_in(t-1)
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # 3. Forward Lookahead (Intent)
    # Shift -1 to -4 to get future values
    df["u_in_lead1"] = df.groupby("breath_id")["u_in"].shift(-1).fillna(0)
    df["u_in_lead2"] = df.groupby("breath_id")["u_in"].shift(-2).fillna(0)
    df["u_in_lead3"] = df.groupby("breath_id")["u_in"].shift(-3).fillna(0)
    df["u_in_lead4"] = df.groupby("breath_id")["u_in"].shift(-4).fillna(0)

    # 4. Volume Integration: sum(u_in * dt)
    # Calculate incremental volume then cumsum
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # 5. Physical Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    # Avoid division by zero if C is 0 (though C is usually > 0), handle gracefully
    df["area_C"] = df["area"] / df["C"]

    # Fill any remaining NaNs (e.g., first row of diffs) with 0
    df = df.fillna(0)

    return df


def get_transformed_data(load_cached_data=True):
    """
    Loads, processes, scales, and reshapes data.
    Returns DataLoaders for train, val, and test sets.
    Implements caching to disk to speed up subsequent runs.
    """
    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", cache_dir)
        train_x = np.load(files["train_x"])
        train_y = np.load(files["train_y"])
        val_x = np.load(files["val_x"])
        val_y = np.load(files["val_y"])
        test_x = np.load(files["test_x"])
        # test_ids are not strictly needed for the dataloader but good to have if needed later
        # We don't return them here to keep signature simple, but they are cached.
    else:
        print("Processing data from scratch...")

        # Load metadata CSVs
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply Feature Engineering
        print("Generating features...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Select Features
        features = Config.FEATURE_COLS
        target = Config.TARGET_COL

        # Scale Data using RobustScaler (excluding u_out)
        # Cite solution_lesson_node_00098: Do not scale binary features intended for logic operations.
        scale_cols = [col for col in features if col != "u_out"]
        print(f"Scaling data (excluding u_out)...")

        scaler = RobustScaler()

        # Fit on training data only
        train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
        # Transform val and test
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Reshape to (N_breaths, 80, N_features)
        # Each breath has exactly 80 time steps
        SEQ_LEN = 80

        print("Reshaping tensors...")
        train_x = train_df[features].values.reshape(-1, SEQ_LEN, len(features))
        train_y = train_df[target].values.reshape(-1, SEQ_LEN)

        val_x = val_df[features].values.reshape(-1, SEQ_LEN, len(features))
        val_y = val_df[target].values.reshape(-1, SEQ_LEN)

        test_x = test_df[features].values.reshape(-1, SEQ_LEN, len(features))
        test_ids = test_df["id"].values  # Flattened IDs for submission mapping

        # Save to cache
        print("Saving to cache...")
        np.save(files["train_x"], train_x)
        np.save(files["train_y"], train_y)
        np.save(files["val_x"], val_x)
        np.save(files["val_y"], val_y)
        np.save(files["test_x"], test_x)
        np.save(files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x)  # No target for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for stability
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

    print(f"Data loaded. Train shape: {train_x.shape}, Val shape: {val_x.shape}")

    return train_loader, val_loader, test_loader

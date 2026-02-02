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
    """

    def __init__(self, X, y=None, u_out=None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, 80, num_features).
            y (np.ndarray, optional): Target pressure of shape (num_breaths, 80).
            u_out (np.ndarray, optional): Control input u_out of shape (num_breaths, 80).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - x: Input features for the breath.
            - u_out: The expiratory valve control signal (used for masking loss).
            - y: The target pressure (if available).
        """
        data = {"x": self.X[idx]}

        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]

        if self.y is not None:
            data["y"] = self.y[idx]

        return data


def prepare_data(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Loads, scales, and reshapes data, then returns PyTorch DataLoaders.
    Implements caching using .npy files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_x": os.path.join(Config.CACHE_DIR, "train_x.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "train_uout": os.path.join(Config.CACHE_DIR, "train_uout.npy"),
        "val_x": os.path.join(Config.CACHE_DIR, "val_x.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "val_uout": os.path.join(Config.CACHE_DIR, "val_uout.npy"),
        "test_x": os.path.join(Config.CACHE_DIR, "test_x.npy"),
        "test_uout": os.path.join(Config.CACHE_DIR, "test_uout.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        train_x = np.load(cache_files["train_x"])
        train_y = np.load(cache_files["train_y"])
        train_uout = np.load(cache_files["train_uout"])

        val_x = np.load(cache_files["val_x"])
        val_y = np.load(cache_files["val_y"])
        val_uout = np.load(cache_files["val_uout"])

        test_x = np.load(cache_files["test_x"])
        test_uout = np.load(cache_files["test_uout"])

    else:
        print("Processing data from scratch...")

        # Load raw CSV data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug Mode: Truncate datasets for faster iteration
        if Config.DEBUG:
            print(f"DEBUG MODE: Using only {Config.DEBUG_BREATHS} breaths per split.")
            train_ids = train_df["breath_id"].unique()[: Config.DEBUG_BREATHS]
            val_ids = val_df["breath_id"].unique()[: Config.DEBUG_BREATHS]
            test_ids = test_df["breath_id"].unique()[: Config.DEBUG_BREATHS]

            train_df = train_df[train_df["breath_id"].isin(train_ids)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_ids)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_ids)].copy()

        # Feature Engineering
        print("Engineering features...")
        for df in [train_df, val_df, test_df]:
            # Group by breath_id to perform lag/cumsum operations
            # Note: The data is already sorted by breath_id and time_step
            grp = df.groupby("breath_id")

            # Lag features
            df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
            df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0)

            # Difference features
            df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
            df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]

            # Cumulative sum (Area) - proxy for volume
            df["area"] = grp["u_in"].cumsum()

        # Feature Scaling
        # We scale continuous features but leave binary u_out alone.
        scale_cols = [
            "time_step",
            "u_in",
            "R",
            "C",
            "area",
            "u_in_lag1",
            "u_in_lag2",
            "u_in_diff1",
            "u_in_diff2",
        ]
        scaler = RobustScaler()

        # Fit scaler only on training data
        scaler.fit(train_df[scale_cols])

        # Transform all splits
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Helper function to reshape flat dataframe to (N, 80, F)
        def reshape_dataset(df, is_test=False):
            # Ensure features are in the correct order defined in Config
            x_flat = df[Config.FEATURE_COLS].values
            u_out_flat = df["u_out"].values

            num_breaths = len(df) // Config.SEQUENCE_LENGTH

            # Reshape inputs
            x = x_flat.reshape(
                num_breaths, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLS)
            )
            u_out = u_out_flat.reshape(num_breaths, Config.SEQUENCE_LENGTH)

            if not is_test:
                y_flat = df[Config.TARGET_COL].values
                y = y_flat.reshape(num_breaths, Config.SEQUENCE_LENGTH)
                return x, y, u_out
            else:
                return x, None, u_out

        # Reshape all datasets
        train_x, train_y, train_uout = reshape_dataset(train_df)
        val_x, val_y, val_uout = reshape_dataset(val_df)
        test_x, _, test_uout = reshape_dataset(test_df, is_test=True)

        # Save to cache
        print(f"Saving processed data to {Config.CACHE_DIR}...")
        np.save(cache_files["train_x"], train_x)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["train_uout"], train_uout)

        np.save(cache_files["val_x"], val_x)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["val_uout"], val_uout)

        np.save(cache_files["test_x"], test_x)
        np.save(cache_files["test_uout"], test_uout)

    # Instantiate Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_uout)
    val_dataset = VentilatorDataset(val_x, val_y, val_uout)
    test_dataset = VentilatorDataset(test_x, None, test_uout)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

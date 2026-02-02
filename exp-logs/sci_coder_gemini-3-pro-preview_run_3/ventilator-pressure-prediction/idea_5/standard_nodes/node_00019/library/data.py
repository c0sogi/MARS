import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    def __init__(self, X, u_out, y=None):
        """
        PyTorch Dataset for Ventilator Pressure Prediction.

        Args:
            X (np.ndarray): Input features of shape (num_breaths, breath_len, num_features).
            u_out (np.ndarray): Control input u_out of shape (num_breaths, breath_len).
                                Used for masking in the loss function.
            y (np.ndarray, optional): Target pressure of shape (num_breaths, breath_len).
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
        if self.y is not None:
            return self.X[idx], self.u_out[idx], self.y[idx]
        else:
            return self.X[idx], self.u_out[idx]


def add_features(df):
    """
    Implements the PID and Physics feature engineering pipeline.
    Generates integral, derivative, and physics-informed interaction terms.
    """
    # Ensure data is sorted by breath_id and time_step to guarantee correct lag/cumsum operations
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # PID Features
    if Config.USE_PID:
        # Integral (Area): Cumulative sum of u_in as a proxy for volume
        df["area"] = df.groupby("breath_id")["u_in"].cumsum()

        # Derivative (Diff): Rate of change of u_in
        df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # Physics Features
    if Config.USE_PHYSICS:
        # Resistive component proxy: Pressure drop ~ R * Flow (u_in)
        df["R_u_in"] = df["R"] * df["u_in"]

        # Elastic component proxy: Pressure ~ Volume / Compliance
        # Ensure 'area' exists if USE_PID was False
        if "area" not in df.columns:
            df["area"] = df.groupby("breath_id")["u_in"].cumsum()
        df["vol_C"] = df["area"] / df["C"]

    return df


def get_dataloaders(load_cached_data=True):
    """
    Main function to load data, process features, and return DataLoaders.
    Handles caching to disk to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files
                                 from Config.WORKING_DIR.
    """
    seed_everything(Config.SEED)

    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "train_u_out": os.path.join(cache_dir, "train_u_out.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "val_u_out": os.path.join(cache_dir, "val_u_out.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_u_out": os.path.join(cache_dir, "test_u_out.npy"),
        "scaler_mean": os.path.join(cache_dir, "scaler_mean.npy"),
        "scaler_scale": os.path.join(cache_dir, "scaler_scale.npy"),
    }

    # Check if we should and can load from cache
    all_exist = all(os.path.exists(f) for f in files.values())

    if load_cached_data and all_exist:
        print(f"Loading cached data from {cache_dir}...")
        train_x = np.load(files["train_x"])
        train_y = np.load(files["train_y"])
        train_u_out = np.load(files["train_u_out"])

        val_x = np.load(files["val_x"])
        val_y = np.load(files["val_y"])
        val_u_out = np.load(files["val_u_out"])

        test_x = np.load(files["test_x"])
        test_u_out = np.load(files["test_u_out"])

    else:
        print("Processing data from scratch...")

        # Load raw CSVs
        print(f"Loading {Config.TRAIN_CSV}...")
        train_df = pd.read_csv(Config.TRAIN_CSV)
        print(f"Loading {Config.VAL_CSV}...")
        val_df = pd.read_csv(Config.VAL_CSV)
        print(f"Loading {Config.TEST_CSV}...")
        test_df = pd.read_csv(Config.TEST_CSV)

        # Debug Mode: Slice data to reduce processing time
        if Config.DEBUG:
            print(f"Debug mode active: using {Config.DEBUG_SIZE} breaths per split.")
            train_breath_ids = train_df["breath_id"].unique()[: Config.DEBUG_SIZE]
            val_breath_ids = val_df["breath_id"].unique()[: Config.DEBUG_SIZE]
            test_breath_ids = test_df["breath_id"].unique()[: Config.DEBUG_SIZE]

            train_df = train_df[train_df["breath_id"].isin(train_breath_ids)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breath_ids)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breath_ids)].copy()

        # Add Features
        print("Generating PID and Physics features...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Define feature columns based on Config
        # Order: time_step, u_in, u_out, R, C, area, u_in_diff, R_u_in, vol_C
        feature_cols = [
            "time_step",
            "u_in",
            "u_out",
            "R",
            "C",
            "area",
            "u_in_diff",
            "R_u_in",
            "vol_C",
        ]

        # Verify dimensions
        if len(feature_cols) != Config.INPUT_DIM:
            raise ValueError(
                f"Feature count ({len(feature_cols)}) does not match Config.INPUT_DIM ({Config.INPUT_DIM})"
            )

        # Normalization
        # We scale all features except u_out (binary).
        cols_to_scale = [c for c in feature_cols if c != "u_out"]

        scaler = StandardScaler()
        print("Fitting scaler on training data...")
        scaler.fit(train_df[cols_to_scale])

        # Save scaler params for reproducibility/inference
        np.save(files["scaler_mean"], scaler.mean_)
        np.save(files["scaler_scale"], scaler.scale_)

        # Transform
        print("Transforming datasets...")
        train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
        test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

        # Helper to reshape data
        def reshape_data(df, is_test=False):
            # Ensure sorting
            df = df.sort_values(["breath_id", "id"])

            # Extract arrays
            x_arr = df[feature_cols].values.astype(np.float32)
            u_out_arr = df["u_out"].values.astype(np.float32)

            # Calculate number of breaths
            num_breaths = len(df) // Config.BREATH_LEN

            # Reshape to (N, 80, Features)
            x_reshaped = x_arr.reshape(num_breaths, Config.BREATH_LEN, -1)
            u_out_reshaped = u_out_arr.reshape(num_breaths, Config.BREATH_LEN)

            if not is_test:
                y_arr = df["pressure"].values.astype(np.float32)
                y_reshaped = y_arr.reshape(num_breaths, Config.BREATH_LEN)
                return x_reshaped, u_out_reshaped, y_reshaped
            else:
                return x_reshaped, u_out_reshaped, None

        print("Reshaping tensors...")
        train_x, train_u_out, train_y = reshape_data(train_df)
        val_x, val_u_out, val_y = reshape_data(val_df)
        test_x, test_u_out, _ = reshape_data(test_df, is_test=True)

        # Save to cache
        print(f"Saving processed data to {cache_dir}...")
        np.save(files["train_x"], train_x)
        np.save(files["train_y"], train_y)
        np.save(files["train_u_out"], train_u_out)

        np.save(files["val_x"], val_x)
        np.save(files["val_y"], val_y)
        np.save(files["val_u_out"], val_u_out)

        np.save(files["test_x"], test_x)
        np.save(files["test_u_out"], test_u_out)

        # Cleanup to free memory
        del train_df, val_df, test_df
        gc.collect()

    # Create Datasets
    print("Creating PyTorch Datasets...")
    train_dataset = VentilatorDataset(train_x, train_u_out, train_y)
    val_dataset = VentilatorDataset(val_x, val_u_out, val_y)
    test_dataset = VentilatorDataset(test_x, test_u_out)

    # Create Loaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader

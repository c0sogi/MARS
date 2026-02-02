import os
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

from library.config import Config, seed_everything
from library.features import FeatureEngineer


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns features, targets, u_out mask, and identifiers.
    """

    def __init__(self, X, u_out, y=None, ids=None, is_test=False):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.u_out = torch.as_tensor(u_out, dtype=torch.float32)

        if y is not None:
            self.y = torch.as_tensor(y, dtype=torch.float32)
        else:
            self.y = None

        if ids is not None:
            self.ids = torch.as_tensor(ids, dtype=torch.int64)
        else:
            self.ids = None

        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx], "u_out": self.u_out[idx]}

        if self.y is not None:
            item["y"] = self.y[idx]

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def get_config_hash():
    """
    Generates a unique hash based on the current data configuration.
    """
    config_dict = {
        "features": sorted(Config.FEATURE_COLS),
        "physics_cols": sorted(Config.PHYSICS_COLS),
        "seq_len": Config.SEQ_LEN,
        "debug": Config.DEBUG,
        "debug_breaths": Config.DEBUG_BREATHS if Config.DEBUG else 0,
        "seed": Config.SEED,
    }
    # Sort keys to ensure consistent ordering
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def save_numpy_cache(data_dict, hash_id, directory):
    """
    Saves a dictionary of numpy arrays to disk with the hash suffix.
    """
    for key, array in data_dict.items():
        filename = f"{key}_{hash_id}.npy"
        path = os.path.join(directory, filename)
        np.save(path, array)


def load_numpy_cache(keys, hash_id, directory):
    """
    Attempts to load numpy arrays from disk. Returns None if any are missing.
    """
    loaded_data = {}
    for key in keys:
        filename = f"{key}_{hash_id}.npy"
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return None
        loaded_data[key] = np.load(path)
    return loaded_data


def reshape_sequences(df, feature_cols, target_col=None, id_col=None):
    """
    Reshapes tabular DataFrame (N_rows, Cols) -> (N_breaths, 80, Cols).
    Assumes data is sorted by breath_id and time_step.
    """
    # Ensure divisible by SEQ_LEN
    assert (
        len(df) % Config.SEQ_LEN == 0
    ), f"Data length {len(df)} not divisible by {Config.SEQ_LEN}"

    num_breaths = len(df) // Config.SEQ_LEN

    # Extract Features
    x_data = df[feature_cols].values.astype(np.float32)
    x_reshaped = x_data.reshape(num_breaths, Config.SEQ_LEN, len(feature_cols))

    # Extract u_out specifically for masking (it is also in x_data usually, but needed separately)
    # Assuming 'u_out' is in the dataframe
    if "u_out" in df.columns:
        u_out_data = df["u_out"].values.astype(np.float32)
        u_out_reshaped = u_out_data.reshape(num_breaths, Config.SEQ_LEN)
    else:
        u_out_reshaped = None

    # Extract Target
    if target_col and target_col in df.columns:
        y_data = df[target_col].values.astype(np.float32)
        y_reshaped = y_data.reshape(num_breaths, Config.SEQ_LEN)
    else:
        y_reshaped = None

    # Extract IDs
    if id_col and id_col in df.columns:
        id_data = df[id_col].values.astype(np.int64)
        id_reshaped = id_data.reshape(num_breaths, Config.SEQ_LEN)
    else:
        id_reshaped = None

    return x_reshaped, u_out_reshaped, y_reshaped, id_reshaped


def get_data_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles feature engineering, caching, normalization, and reshaping.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    hash_id = get_config_hash()
    print(f"Data Pipeline Hash: {hash_id}")

    # Define cache keys
    cache_keys = [
        "train_x",
        "train_uout",
        "train_y",
        "val_x",
        "val_uout",
        "val_y",
        "test_x",
        "test_uout",
        "test_ids",
        "scaler_center",
        "scaler_scale",
    ]

    # 1. Try Loading Cache
    cached_data = None
    if load_cached_data:
        print("Checking for cached data...")
        cached_data = load_numpy_cache(cache_keys, hash_id, Config.WORKING_DIR)

    if cached_data is not None:
        print("Loaded data from cache.")
        train_x = cached_data["train_x"]
        train_uout = cached_data["train_uout"]
        train_y = cached_data["train_y"]
        val_x = cached_data["val_x"]
        val_uout = cached_data["val_uout"]
        val_y = cached_data["val_y"]
        test_x = cached_data["test_x"]
        test_uout = cached_data["test_uout"]
        test_ids = cached_data["test_ids"]
    else:
        print("Cache miss or force reload. Processing data from scratch...")

        # Initialize Feature Engineer
        fe = FeatureEngineer()

        # Process Splits
        # Note: fe.process_split handles its own parquet caching for the tabular stage
        train_df = fe.process_split("train", load_cached_data=load_cached_data)
        val_df = fe.process_split("val", load_cached_data=load_cached_data)
        test_df = fe.process_split("test", load_cached_data=load_cached_data)

        print("Reshaping and Normalizing...")

        # Prepare Scaler
        # Fit scaler on flattened training features
        scaler = RobustScaler()
        train_feats_flat = train_df[Config.FEATURE_COLS].values
        scaler.fit(train_feats_flat)

        # Save Scaler params (No pickle)
        scaler_center = scaler.center_
        scaler_scale = scaler.scale_

        # Helper to transform and reshape
        def process_df(df, is_train=True):
            # Transform features
            # Note: We transform the whole tabular DF before reshaping
            # This is efficient
            x_values = df[Config.FEATURE_COLS].values
            # Manual application of RobustScaler to avoid re-instantiating object
            x_scaled = (x_values - scaler_center) / scaler_scale

            # Replace columns in a copy or new dataframe for reshaping
            # To save memory, we can just pass the scaled array to a custom reshaper
            # but we need to keep u_out unscaled for the mask.

            # Re-construct a temporary structure for reshaping
            # Actually, we can just reshape the scaled array directly
            num_breaths = len(df) // Config.SEQ_LEN
            x_reshaped = x_scaled.reshape(
                num_breaths, Config.SEQ_LEN, len(Config.FEATURE_COLS)
            )

            # Get u_out (unscaled)
            u_out_reshaped = df["u_out"].values.reshape(num_breaths, Config.SEQ_LEN)

            # Get Target
            y_reshaped = None
            if "pressure" in df.columns:
                y_reshaped = df["pressure"].values.reshape(num_breaths, Config.SEQ_LEN)

            # Get IDs
            id_reshaped = df["id"].values.reshape(num_breaths, Config.SEQ_LEN)

            return x_reshaped, u_out_reshaped, y_reshaped, id_reshaped

        # Apply processing
        train_x, train_uout, train_y, _ = process_df(train_df, is_train=True)
        val_x, val_uout, val_y, _ = process_df(val_df, is_train=False)
        test_x, test_uout, _, test_ids = process_df(test_df, is_train=False)

        # Save to Cache
        print("Saving processed data to cache...")
        cache_payload = {
            "train_x": train_x,
            "train_uout": train_uout,
            "train_y": train_y,
            "val_x": val_x,
            "val_uout": val_uout,
            "val_y": val_y,
            "test_x": test_x,
            "test_uout": test_uout,
            "test_ids": test_ids,
            "scaler_center": scaler_center,
            "scaler_scale": scaler_scale,
        }
        save_numpy_cache(cache_payload, hash_id, Config.WORKING_DIR)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_uout, train_y)
    val_dataset = VentilatorDataset(val_x, val_uout, val_y)
    test_dataset = VentilatorDataset(test_x, test_uout, ids=test_ids, is_test=True)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    print(f"Data Loaders Ready.")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader

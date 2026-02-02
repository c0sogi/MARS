import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything()


def add_physics_features(df):
    """
    Adds physics-inspired features to the dataframe based on the Equation of Motion.
    """
    # Calculate cumulative sum of u_in per breath (proxy for volume)
    # Groupby is necessary to ensure cumsum resets for each breath
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # R_flow: Resistance * Flow (u_in)
    # Represents the resistive component of pressure
    df["R_flow"] = df["R"] * df["u_in"]

    # C_volume: Volume / Compliance
    # Represents the elastic component of pressure
    df["C_volume"] = df["u_in_cumsum"] / df["C"]

    # Cite solution_lesson_node_00003: Explicit Derivative Features
    # Lag features
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)

    # Finite difference features (Derivative proxy)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - (
        df.groupby("breath_id")["u_in_diff1"].shift(1).fillna(0)
    )

    return df


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, 80, N_features)
            y (np.ndarray, optional): Target pressure of shape (N_breaths, 80)
            is_test (bool): Whether this is the test set (no targets)
        """
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to float32 tensors
        x_item = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            return x_item

        y_item = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_item, y_item


def load_and_process_data(load_cached_data=True):
    """
    Loads data, generates features, normalizes, and reshapes.
    Handles caching to speed up subsequent runs.

    Returns:
        train_X, train_y, val_X, val_y, test_X (np.ndarrays)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_files = [
        Config.TRAIN_CACHE_DATA,
        Config.TRAIN_CACHE_TARGET,
        Config.VAL_CACHE_DATA,
        Config.VAL_CACHE_TARGET,
        Config.TEST_CACHE_DATA,
        Config.STATS_CACHE,
    ]
    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_X = np.load(Config.TRAIN_CACHE_DATA)
        train_y = np.load(Config.TRAIN_CACHE_TARGET)
        val_X = np.load(Config.VAL_CACHE_DATA)
        val_y = np.load(Config.VAL_CACHE_TARGET)
        test_X = np.load(Config.TEST_CACHE_DATA)
        return train_X, train_y, val_X, val_y, test_X

    print("Processing data from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Debug mode: sample a subset of breaths for rapid testing
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
        train_breaths = train_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_breaths = val_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        test_breaths = test_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]

        train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
        val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
        test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

    # Feature Engineering
    print("Generating physics features...")
    train_df = add_physics_features(train_df)
    val_df = add_physics_features(val_df)
    test_df = add_physics_features(test_df)

    # Select features defined in Config
    feature_cols = Config.FEATURE_COLS

    # Extract values
    train_features = train_df[feature_cols].values.astype(np.float32)
    val_features = val_df[feature_cols].values.astype(np.float32)
    test_features = test_df[feature_cols].values.astype(np.float32)

    # Normalization
    # Compute stats on TRAIN set only to prevent leakage
    print("Normalizing features...")
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    # Save stats as a stacked array (2, n_features) to avoid pickle
    stats = np.stack([mean, std])
    np.save(Config.STATS_CACHE, stats)

    # Apply normalization
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std

    # Reshape to (N_breaths, 80, N_features)
    # The data is already grouped by breath_id in the metadata generation
    print("Reshaping data...")
    train_X = train_features.reshape(-1, Config.SEQ_LEN, len(feature_cols))
    val_X = val_features.reshape(-1, Config.SEQ_LEN, len(feature_cols))
    test_X = test_features.reshape(-1, Config.SEQ_LEN, len(feature_cols))

    # Prepare targets
    train_y = train_df["pressure"].values.astype(np.float32).reshape(-1, Config.SEQ_LEN)
    val_y = val_df["pressure"].values.astype(np.float32).reshape(-1, Config.SEQ_LEN)

    # Save to cache
    print("Saving data to cache...")
    np.save(Config.TRAIN_CACHE_DATA, train_X)
    np.save(Config.TRAIN_CACHE_TARGET, train_y)
    np.save(Config.VAL_CACHE_DATA, val_X)
    np.save(Config.VAL_CACHE_TARGET, val_y)
    np.save(Config.TEST_CACHE_DATA, test_X)

    return train_X, train_y, val_X, val_y, test_X


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_X, train_y, val_X, val_y, test_X = load_and_process_data(load_cached_data)

    train_dataset = VentilatorDataset(train_X, train_y, is_test=False)
    val_dataset = VentilatorDataset(val_X, val_y, is_test=False)
    test_dataset = VentilatorDataset(test_X, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader

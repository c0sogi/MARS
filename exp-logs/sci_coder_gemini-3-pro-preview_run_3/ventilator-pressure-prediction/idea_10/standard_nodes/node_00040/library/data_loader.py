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

    Attributes:
        X (np.ndarray): Input features of shape (Batch, Seq_Len, Features).
        y (np.ndarray): Target pressure of shape (Batch, Seq_Len, 1). Optional.
        u_out (np.ndarray): Expiratory phase mask of shape (Batch, Seq_Len). Optional.
    """

    def __init__(self, X, y=None, u_out=None):
        self.X = X
        self.y = y
        self.u_out = u_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": torch.tensor(self.X[idx], dtype=torch.float32)}
        if self.y is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.u_out is not None:
            item["u_out"] = torch.tensor(self.u_out[idx], dtype=torch.float32)
        return item


def add_features(df):
    """
    Computes dynamic PID features and physics interaction terms.

    Args:
        df (pd.DataFrame): Raw dataframe containing breath sequences.

    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    # Group by breath_id to ensure calculations respect breath boundaries
    gb = df.groupby(Config.BREATH_ID_COL)

    # 1. PID State: Integral (Volume Proxy)
    # Cumulative sum of u_in approximates volume
    df["u_in_cumsum"] = gb["u_in"].cumsum()

    # 2. PID State: Derivative (Flow Proxy)
    # First difference of u_in
    df["u_in_diff1"] = gb["u_in"].diff().fillna(0)

    # 3. PID State: Acceleration
    # Second difference (difference of the derivative)
    # We must group again on the modified dataframe or use the cached grouper carefully.
    # Grouping again is safer to handle the new column.
    df["u_in_diff2"] = df.groupby(Config.BREATH_ID_COL)["u_in_diff1"].diff().fillna(0)

    # 4. Physics Interactions
    # Resistive Pressure component: R * Flow (approximated by u_in)
    df["R_u_in"] = df["R"] * df["u_in"]

    # Elastic Pressure component: Volume / C
    # Avoid division by zero issues if C were 0 (not the case here, but good practice)
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    return df


def get_data_loaders(debug=Config.DEBUG):
    """
    Orchestrates data loading, feature engineering, scaling, and caching.

    Args:
        debug (bool): If True, uses a small subset of data for quick testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define auxiliary cache paths for u_out (needed for masking)
    cache_train_u_out = os.path.join(Config.WORKING_DIR, "train_u_out.npy")
    cache_val_u_out = os.path.join(Config.WORKING_DIR, "val_u_out.npy")
    cache_test_u_out = os.path.join(Config.WORKING_DIR, "test_u_out.npy")

    # Check if all required cache files exist
    required_cache = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        cache_train_u_out,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        cache_val_u_out,
        Config.CACHE_TEST_X,
        cache_test_u_out,
        Config.CACHE_TEST_IDS,
        Config.CACHE_SCALER,
    ]
    cache_exists = all(os.path.exists(f) for f in required_cache)

    if Config.FORCE_REGENERATE_CACHE or not cache_exists:
        print("Cache not found or regeneration forced. Processing data...")

        # 1. Load Metadata
        print(f"Loading data from {Config.METADATA_DIR}...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        if debug:
            print(f"DEBUG MODE: Subsampling first {Config.DEBUG_SAMPLES} breaths...")
            # Assuming ~80 steps per breath
            subset_rows = Config.DEBUG_SAMPLES * Config.MAX_SEQ_LEN
            train_df = train_df.iloc[:subset_rows].copy()
            val_df = val_df.iloc[:subset_rows].copy()
            test_df = test_df.iloc[:subset_rows].copy()

        # 2. Feature Engineering
        print("Generating features...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # 3. Robust Scaling
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        scaler.fit(train_df[Config.FEATURE_COLS])

        # Save scaler statistics for inference/analysis
        np.savez(Config.CACHE_SCALER, center=scaler.center_, scale=scaler.scale_)

        # 4. Processing & Reshaping Helper
        def process_split(df, is_test=False):
            # Transform features
            x_scaled = scaler.transform(df[Config.FEATURE_COLS])

            # Determine dimensions
            n_rows = len(df)
            n_breaths = n_rows // Config.MAX_SEQ_LEN

            # Handle potential edge cases where rows aren't perfectly divisible
            # (Though dataset is usually clean)
            if n_rows % Config.MAX_SEQ_LEN != 0:
                trim = n_breaths * Config.MAX_SEQ_LEN
                x_scaled = x_scaled[:trim]
                df = df.iloc[:trim]

            # Reshape to (Batch, Seq, Features)
            x = x_scaled.reshape(
                n_breaths, Config.MAX_SEQ_LEN, len(Config.FEATURE_COLS)
            )

            # Extract u_out for masking (Batch, Seq)
            u_out = df["u_out"].values.reshape(n_breaths, Config.MAX_SEQ_LEN)

            y = None
            if not is_test:
                y = df[Config.TARGET_COL].values.reshape(
                    n_breaths, Config.MAX_SEQ_LEN, 1
                )

            ids = None
            if is_test:
                ids = df[Config.ID_COL].values  # Flat IDs for submission

            return x, y, u_out, ids

        print("Reshaping and creating tensors...")
        train_x, train_y, train_u_out, _ = process_split(train_df)
        val_x, val_y, val_u_out, _ = process_split(val_df)
        test_x, _, test_u_out, test_ids = process_split(test_df, is_test=True)

        # 5. Save to Cache (Skip if debugging to avoid corrupting main cache)
        if not debug:
            print(f"Saving processed arrays to {Config.WORKING_DIR}...")
            np.save(Config.CACHE_TRAIN_X, train_x)
            np.save(Config.CACHE_TRAIN_Y, train_y)
            np.save(cache_train_u_out, train_u_out)

            np.save(Config.CACHE_VAL_X, val_x)
            np.save(Config.CACHE_VAL_Y, val_y)
            np.save(cache_val_u_out, val_u_out)

            np.save(Config.CACHE_TEST_X, test_x)
            np.save(cache_test_u_out, test_u_out)
            np.save(Config.CACHE_TEST_IDS, test_ids)
        else:
            print("Debug mode: Skipping cache save.")

    else:
        print("Loading data from cache...")
        train_x = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        train_u_out = np.load(cache_train_u_out)

        val_x = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        val_u_out = np.load(cache_val_u_out)

        test_x = np.load(Config.CACHE_TEST_X)
        test_u_out = np.load(cache_test_u_out)
        # test_ids are loaded by the submission script usually, but available if needed

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_x, val_y, val_u_out)
    test_dataset = VentilatorDataset(test_x, None, test_u_out)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

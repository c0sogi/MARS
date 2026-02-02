import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, seq_len, num_features).
            y (np.ndarray, optional): Target pressure of shape (num_breaths, seq_len).
            is_test (bool): Whether this is the test set (returns only X).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def add_features(df):
    """
    Adds engineering features to the dataframe.
    - u_in_cumsum: Cumulative sum of u_in per breath.
    """
    # Ensure sorted by breath_id and time_step for correct cumsum
    # (Data is usually sorted, but safety first)
    # Note: We assume the incoming df is already sorted or grouped appropriately if needed,
    # but raw train.csv is sorted by id.

    # Group by breath_id and calculate cumsum of u_in
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # Add physics-based interaction features (Cite solution_lesson_node_00004)
    # Resistive pressure component
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic pressure component
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    return df


def get_manual_scaler_params(df, columns):
    """
    Fits a RobustScaler and returns the center and scale parameters.
    Used to avoid pickling the scaler object.
    """
    scaler = RobustScaler()
    scaler.fit(df[columns])
    return scaler.center_, scaler.scale_


def apply_manual_scaler(df, columns, center, scale):
    """
    Manually applies RobustScaler transformation: (X - center) / scale
    """
    # Avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)

    X = df[columns].values
    X_scaled = (X - center) / scale

    df_scaled = df.copy()
    df_scaled[columns] = X_scaled
    return df_scaled


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches data.

    Logic:
    1. Check if cached .npz files exist.
    2. If load_cached_data=True and files exist, load and return.
    3. Else, process raw CSVs:
       - Split train.csv based on metadata.
       - Feature engineering (u_in_cumsum).
       - Fit scaler on Train, transform Train/Val/Test.
       - Reshape to (N, 80, Features).
       - Save to cache.
    """
    # Define cache paths (using .npz for array storage)
    train_path = os.path.join(Config.CACHE_DIR, "train_data.npz")
    val_path = os.path.join(Config.CACHE_DIR, "val_data.npz")
    test_path = os.path.join(Config.CACHE_DIR, "test_data.npz")
    scaler_path = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
        and os.path.exists(scaler_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from", Config.CACHE_DIR)
        train_data = np.load(train_path)
        val_data = np.load(val_path)
        test_data = np.load(test_path)

        # Return dictionaries/arrays
        return (
            train_data["X"],
            train_data["y"],
            val_data["X"],
            val_data["y"],
            test_data["X"],
            test_data["ids"],
        )

    print("Processing data from scratch...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --- 1. Load Metadata ---
    print("Loading metadata...")
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)

    train_breath_ids = train_meta["breath_id"].unique()
    val_breath_ids = val_meta["breath_id"].unique()

    # --- 2. Load and Split Raw Training Data ---
    print(f"Loading raw train data from {Config.TRAIN_CSV}...")
    df_full_train = pd.read_csv(Config.TRAIN_CSV)

    # Debugging: Sample breaths if DEBUG is on
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
        all_breaths = df_full_train["breath_id"].unique()
        sample_breaths = all_breaths[: Config.DEBUG_SAMPLE_SIZE]
        df_full_train = df_full_train[
            df_full_train["breath_id"].isin(sample_breaths)
        ].copy()

        # Filter metadata IDs to match sample for consistency
        train_breath_ids = [b for b in train_breath_ids if b in sample_breaths]
        val_breath_ids = [b for b in val_breath_ids if b in sample_breaths]

    # Split
    df_train = df_full_train[df_full_train["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_full_train[df_full_train["breath_id"].isin(val_breath_ids)].copy()

    del df_full_train  # Free memory

    # --- 3. Feature Engineering & Scaling ---
    print("Feature engineering...")
    df_train = add_features(df_train)
    df_val = add_features(df_val)

    # Columns configuration
    # Input Features: [time_step, u_in, u_out, R, C, u_in_cumsum, R_u_in, vol_C]
    scale_cols = ["time_step", "u_in", "R", "C", "u_in_cumsum", "R_u_in", "vol_C"]
    feature_order = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
    ]

    print("Fitting scaler on training data...")
    center, scale = get_manual_scaler_params(df_train, scale_cols)

    # Save scaler params
    np.savez(scaler_path, center=center, scale=scale)

    print("Transforming data...")
    df_train = apply_manual_scaler(df_train, scale_cols, center, scale)
    df_val = apply_manual_scaler(df_val, scale_cols, center, scale)

    # --- 4. Reshaping and Saving Train/Val ---
    def reshape_dataset(df, is_test=False):
        # Ensure sorting
        df = df.sort_values(["breath_id", "id"])

        # Extract features
        X = df[feature_order].values.astype(np.float32)

        # Reshape: (Num Breaths, 80, Num Features)
        num_breaths = len(df) // Config.SEQ_LEN
        X = X.reshape(num_breaths, Config.SEQ_LEN, len(feature_order))

        if is_test:
            ids = df["id"].values.astype(np.int32)
            return X, ids
        else:
            y = df["pressure"].values.astype(np.float32)
            y = y.reshape(num_breaths, Config.SEQ_LEN)
            return X, y

    print("Reshaping train/val datasets...")
    X_train, y_train = reshape_dataset(df_train)
    X_val, y_val = reshape_dataset(df_val)

    print(f"Saving train data to {train_path}...")
    np.savez(train_path, X=X_train, y=y_train)

    print(f"Saving val data to {val_path}...")
    np.savez(val_path, X=X_val, y=y_val)

    # Free memory
    del df_train, df_val, X_train, y_train, X_val, y_val

    # --- 5. Process Test Data ---
    print(f"Loading raw test data from {Config.TEST_CSV}...")
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        test_breaths = df_test["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test[df_test["breath_id"].isin(test_breaths)].copy()

    print("Processing test data...")
    df_test = add_features(df_test)
    df_test = apply_manual_scaler(df_test, scale_cols, center, scale)

    X_test, test_ids = reshape_dataset(df_test, is_test=True)

    print(f"Saving test data to {test_path}...")
    np.savez(test_path, X=X_test, ids=test_ids)

    return load_and_process_data(
        split="all", load_cached_data=True
    )  # Reload from cache to ensure consistency


def load_and_process_data(split="all", load_cached_data=True):
    """
    Helper to call prepare_data and return specific splits if needed.
    Currently prepare_data handles the orchestration.
    """
    # This is a wrapper to satisfy the pattern if called recursively or externally
    train_path = os.path.join(Config.CACHE_DIR, "train_data.npz")

    if not os.path.exists(train_path) or not load_cached_data:
        return prepare_data(load_cached_data)

    # Load
    train_data = np.load(os.path.join(Config.CACHE_DIR, "train_data.npz"))
    val_data = np.load(os.path.join(Config.CACHE_DIR, "val_data.npz"))
    test_data = np.load(os.path.join(Config.CACHE_DIR, "test_data.npz"))

    return (
        train_data["X"],
        train_data["y"],
        val_data["X"],
        val_data["y"],
        test_data["X"],
        test_data["ids"],
    )


def get_data_loaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    seed_everything(Config.SEED)

    X_train, y_train, X_val, y_val, X_test, test_ids = prepare_data(load_cached_data)

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Val Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test, is_test=True)

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

    return train_loader, val_loader, test_loader, test_ids

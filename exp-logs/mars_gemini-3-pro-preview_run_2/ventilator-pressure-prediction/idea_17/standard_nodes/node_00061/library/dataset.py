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
    Groups time steps into breaths (sequences of length 80).
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, 80, num_features).
            u_out (np.ndarray): Expiratory valve control of shape (num_breaths, 80).
            y (np.ndarray, optional): Target pressure of shape (num_breaths, 80).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.u_out[idx], self.y[idx]
        return self.X[idx], self.u_out[idx]


def add_features(df):
    """
    Applies physics-based feature engineering and dynamics extraction.
    Strictly excludes future lags.
    """
    # Ensure data is sorted by breath_id and time_step (should be already, but safety first)
    df = df.sort_values([Config.BREATH_COL, Config.TIME_COL])

    # 1. Time Delta and Volume Integration
    # dt = t[i] - t[i-1]
    df["dt"] = df.groupby(Config.BREATH_COL)[Config.TIME_COL].diff().fillna(0)
    # volume = cumsum(u_in * dt)
    # Note: u_in is 0-100, dt is in seconds.
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df[Config.BREATH_COL]).cumsum()

    # 2. Physics Interaction Terms
    # Pressure ~ R * Flow + Volume / C
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["volume"] / df["C"]

    # 3. Explicit Dynamics (Past Lags)
    # Lags: 1 to 4
    for lag in Config.LAGS:
        df[f"u_in_lag{lag}"] = (
            df.groupby(Config.BREATH_COL)["u_in"].shift(lag).fillna(0)
        )

    # 4. Finite Differences (Derivatives)
    # 1st Derivative: diff1 = u_in[t] - u_in[t-1]
    df["u_in_diff1"] = df.groupby(Config.BREATH_COL)["u_in"].diff(1).fillna(0)

    # 2nd Derivative: diff2 = diff1[t] - diff1[t-1]
    # This approximates acceleration of the valve
    df["u_in_diff2"] = df.groupby(Config.BREATH_COL)["u_in_diff1"].diff(1).fillna(0)

    # R and C are already continuous. u_out is binary.
    return df


def load_and_preprocess_data(split, load_cached_data=True):
    """
    Loads data, generates features, scales, and reshapes into sequences.
    Handles caching to disk.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, u_out, y) arrays. y is None for test split.
    """
    # Define cache paths
    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_processed.npz")
    scaler_path = Config.SCALER_PATH

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache: {cache_file}")
        data = np.load(cache_file)
        # Check if y exists in the archive
        y = data["y"] if "y" in data else None
        return data["X"], data["u_out"], y

    print(f"Processing {split} data from scratch...")

    # 2. Load Metadata and Raw Data
    if split == "train":
        meta_path = Config.TRAIN_META
        raw_path = Config.TRAIN_CSV
    elif split == "val":
        meta_path = Config.VAL_META
        raw_path = Config.TRAIN_CSV  # Val comes from train.csv
    elif split == "test":
        meta_path = Config.TEST_META
        raw_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load metadata
    df_meta = pd.read_csv(meta_path)
    breath_ids = df_meta[Config.BREATH_COL].unique()

    # Debug mode: subsample breaths
    if Config.DEBUG:
        print(f"DEBUG MODE: limiting to {Config.DEBUG_BREATHS} breaths.")
        breath_ids = breath_ids[: Config.DEBUG_BREATHS]
        df_meta = df_meta[df_meta[Config.BREATH_COL].isin(breath_ids)]

    # Load raw csv
    # Optimization: Load full CSV then filter is often faster than chunking for this size
    df_raw = pd.read_csv(raw_path)
    df = df_raw[df_raw[Config.BREATH_COL].isin(breath_ids)].copy()

    # 3. Feature Engineering
    df = add_features(df)

    # Define feature columns
    # Base continuous columns + generated ones
    cols_cont = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_diff1",
        "u_in_diff2",
    ] + [f"u_in_lag{lag}" for lag in Config.LAGS]

    # 4. Scaling (RobustScaler)
    # We manually handle scaler state to avoid pickling
    if split == "train":
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler(quantile_range=(Config.ROBUST_Q_MIN, Config.ROBUST_Q_MAX))
        scaler.fit(df[cols_cont].values)

        # Save scaler params
        np.savez(scaler_path, center=scaler.center_, scale=scaler.scale_)
        print(f"Scaler parameters saved to {scaler_path}")

        X_cont = scaler.transform(df[cols_cont].values)
    else:
        # Load scaler params
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler params not found at {scaler_path}. Run train split first."
            )

        print(f"Loading scaler parameters from {scaler_path}...")
        params = np.load(scaler_path)

        # Reconstruct scaler
        scaler = RobustScaler(quantile_range=(Config.ROBUST_Q_MIN, Config.ROBUST_Q_MAX))
        scaler.center_ = params["center"]
        scaler.scale_ = params["scale"]

        X_cont = scaler.transform(df[cols_cont].values)

    # 5. Assemble X and Reshape
    # X includes scaled continuous features AND u_out (raw binary)
    # u_out is also returned separately for masking
    u_out_col = df["u_out"].values.reshape(-1, 1)
    X_all = np.hstack([X_cont, u_out_col])

    # Reshape to (num_breaths, 80, num_features)
    num_breaths = len(breath_ids)
    steps_per_breath = 80
    num_features = X_all.shape[1]

    # Ensure data is sorted by breath_id then time_step before reshape
    # (It was sorted in add_features, but df order is preserved)
    X_reshaped = X_all.reshape(num_breaths, steps_per_breath, num_features)
    u_out_reshaped = df["u_out"].values.reshape(num_breaths, steps_per_breath)

    if Config.TARGET_COL in df.columns:
        y_reshaped = df[Config.TARGET_COL].values.reshape(num_breaths, steps_per_breath)
    else:
        y_reshaped = None

    # 6. Save to Cache
    print(f"Saving processed {split} data to {cache_file}...")
    save_dict = {"X": X_reshaped, "u_out": u_out_reshaped}
    if y_reshaped is not None:
        save_dict["y"] = y_reshaped

    np.savez_compressed(cache_file, **save_dict)

    return X_reshaped, u_out_reshaped, y_reshaped


def get_data_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Process Train (fits scaler)
    X_train, u_out_train, y_train = load_and_preprocess_data("train", load_cached_data)
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Process Val (uses scaler)
    X_val, u_out_val, y_val = load_and_preprocess_data("val", load_cached_data)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    X_test, u_out_test, _ = load_and_preprocess_data("test", load_cached_data)
    test_dataset = VentilatorDataset(X_test, u_out_test, y=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader

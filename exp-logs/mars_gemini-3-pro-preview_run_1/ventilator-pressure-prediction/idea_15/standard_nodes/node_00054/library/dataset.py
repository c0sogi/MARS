import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import FeatureEngineer
from library.utils import ensure_dir


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Stores pre-processed tensors for features, masks, and targets.
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, 80, N_features).
            u_out (np.ndarray): Control input u_out of shape (N_breaths, 80).
                                Used for masking loss (0 = Inspiratory, 1 = Expiratory).
            y (np.ndarray, optional): Target pressure of shape (N_breaths, 80).
        """
        self.X = torch.FloatTensor(X)
        self.u_out = torch.FloatTensor(u_out)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "input": self.X[idx],
            "u_out": self.u_out[idx],
        }
        if self.y is not None:
            item["target"] = self.y[idx]
        return item


def prepare_tensors(
    config: Config,
    df: pd.DataFrame,
    split_name: str,
    scaler_center: np.ndarray,
    scaler_scale: np.ndarray,
    debug: bool = False,
):
    """
    Prepares tensors for a specific split:
    1. Checks for cached .npy files.
    2. If not found, extracts features, scales, reshapes, and saves to cache.
    3. Returns numpy arrays (X, u_out, y).
    """

    # Define cache paths
    suffix = "_debug" if debug else ""
    cache_X = os.path.join(config.WORKING_DIR, f"{split_name}_X{suffix}.npy")
    cache_u_out = os.path.join(config.WORKING_DIR, f"{split_name}_u_out{suffix}.npy")
    cache_y = os.path.join(config.WORKING_DIR, f"{split_name}_y{suffix}.npy")

    # Check if cache exists
    # Note: For test set, y might not exist, so we check X and u_out
    has_y = config.TARGET_COL in df.columns
    cache_exists = os.path.exists(cache_X) and os.path.exists(cache_u_out)
    if has_y:
        cache_exists = cache_exists and os.path.exists(cache_y)

    if cache_exists:
        print(f"Loading {split_name} tensors from cache...")
        X = np.load(cache_X)
        u_out = np.load(cache_u_out)
        y = np.load(cache_y) if has_y else None
        return X, u_out, y

    print(f"Processing {split_name} tensors...")

    # 1. Extract Features and Scale
    # We assume FeatureEngineer has already ensured all columns in FEATURE_LIST exist
    features = df[config.FEATURE_LIST].values.astype(np.float32)

    # Apply RobustScaler
    # (X - Median) / IQR
    features = (features - scaler_center) / scaler_scale

    # 2. Reshape
    # Ensure divisible by SEQ_LEN (80)
    num_rows = features.shape[0]
    if num_rows % config.SEQ_LEN != 0:
        raise ValueError(
            f"Data length {num_rows} is not divisible by sequence length {config.SEQ_LEN}"
        )

    num_breaths = num_rows // config.SEQ_LEN
    num_features = features.shape[1]

    X = features.reshape(num_breaths, config.SEQ_LEN, num_features)

    # 3. Extract u_out (Raw) for masking
    # u_out is also in features (scaled), but we need raw 0/1 for loss masking
    u_out_raw = (
        df["u_out"].values.astype(np.float32).reshape(num_breaths, config.SEQ_LEN)
    )

    # 4. Extract Target if available
    y = None
    if has_y:
        y = (
            df[config.TARGET_COL]
            .values.astype(np.float32)
            .reshape(num_breaths, config.SEQ_LEN)
        )

    # 5. Save to Cache
    print(f"Saving {split_name} tensors to {config.WORKING_DIR}...")
    ensure_dir(cache_X)
    np.save(cache_X, X)
    np.save(cache_u_out, u_out_raw)
    if y is not None:
        np.save(cache_y, y)

    return X, u_out_raw, y


def get_data_loaders(config: Config, debug: bool = False):
    """
    Main entry point to get DataLoaders.

    Args:
        config (Config): Configuration object.
        debug (bool): If True, runs in debug mode (fewer epochs, less data).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Feature Engineering (Load or Compute DataFrames)
    fe = FeatureEngineer(config)
    train_df, val_df, test_df = fe.run(load_cached_data=True, debug=debug)

    # 2. Load Scaler Statistics
    # These are computed by FeatureEngineer.run() on the training set
    if not os.path.exists(config.SCALER_CENTER) or not os.path.exists(
        config.SCALER_SCALE
    ):
        raise FileNotFoundError(
            "Scaler statistics not found. FeatureEngineer should have generated them."
        )

    scaler_center = np.load(config.SCALER_CENTER)
    scaler_scale = np.load(config.SCALER_SCALE)

    # 3. Prepare Tensors (Scale, Reshape, Cache)
    train_X, train_uout, train_y = prepare_tensors(
        config, train_df, "train", scaler_center, scaler_scale, debug
    )
    val_X, val_uout, val_y = prepare_tensors(
        config, val_df, "val", scaler_center, scaler_scale, debug
    )
    test_X, test_uout, _ = prepare_tensors(
        config, test_df, "test", scaler_center, scaler_scale, debug
    )

    # 4. Create Datasets
    train_dataset = VentilatorDataset(train_X, train_uout, train_y)
    val_dataset = VentilatorDataset(val_X, val_uout, val_y)
    test_dataset = VentilatorDataset(test_X, test_uout, y=None)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop incomplete batches to maintain shape consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader

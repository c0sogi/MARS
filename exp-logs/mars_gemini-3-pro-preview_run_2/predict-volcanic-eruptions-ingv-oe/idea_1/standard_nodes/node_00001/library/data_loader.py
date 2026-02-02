import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import random

# Import from provided libraries
from library.config import Config
from library.feature_extractor import extract_features


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class VolcanoDataset(Dataset):
    """
    Custom Dataset for Volcano Seismic Data.
    Wraps feature tensors and targets.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (np.ndarray): Array of feature vectors.
            targets (np.ndarray, optional): Array of target values.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


def prepare_data(
    debug_size=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Orchestrates data preparation:
    1. Loads features using feature_extractor (cached if available).
    2. Fits StandardScaler on training data.
    3. Transforms train and val data.
    4. Saves scaler parameters.
    5. Returns DataLoaders.

    Args:
        debug_size (int, optional): Number of samples to load for debugging.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        tuple: (train_loader, val_loader, scaler, input_dim)
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Features
    # The extract_features function handles the parquet caching logic internally based on file existence.
    # We pass the cache paths from Config.

    print("Preparing Training Data...")
    df_train = extract_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_CACHE,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    print("Preparing Validation Data...")
    df_val = extract_features(
        Config.VAL_METADATA_PATH,
        Config.VAL_FEATURES_CACHE,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    # 2. Separate Features and Targets
    # Exclude non-feature columns
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["time_to_eruption"].values.astype(np.float32)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["time_to_eruption"].values.astype(np.float32)

    input_dim = len(feature_cols)

    # 3. Scaling
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Save Scaler Parameters
    # We save mean and scale to npy files to avoid pickle and for inference use later
    mean_path = os.path.join(Config.WORKING_DIR, "scaler_mean.npy")
    scale_path = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")

    np.save(mean_path, scaler.mean_)
    np.save(scale_path, scaler.scale_)
    print(f"Scaler parameters saved to {Config.WORKING_DIR}")

    # 5. Create Datasets and DataLoaders
    train_dataset = VolcanoDataset(X_train_scaled, y_train)
    val_dataset = VolcanoDataset(X_val_scaled, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, scaler, input_dim

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.features import prepare_dataset


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Wraps pre-processed tensors for training, validation, and testing.
    """

    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (torch.Tensor): Input features of shape (N, Seq_Len, Features).
            y (torch.Tensor, optional): Targets of shape (N, Seq_Len).
            is_test (bool): If True, __getitem__ returns only X.
        """
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Orchestrates the data pipeline:
    1. Manages cache invalidation.
    2. Loads features using library.features.prepare_dataset.
    3. Performs RobustScaling (excluding binary masks).
    4. Returns PyTorch DataLoaders.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If False, deletes existing cache and recomputes.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure environment is set up
    Config.initialize()

    # --- 1. Cache Invalidation ---
    if not load_cached_data:
        print("Cache invalidation requested. Removing stale .npy files...")
        cache_bases = [Config.TRAIN_CACHE, Config.VAL_CACHE, Config.TEST_CACHE]
        for base in cache_bases:
            # features.py appends _x.npy and _y.npy
            x_path = base.replace(".npy", "_x.npy")
            y_path = base.replace(".npy", "_y.npy")

            if os.path.exists(x_path):
                os.remove(x_path)
            if os.path.exists(y_path):
                os.remove(y_path)

    # --- 2. Load Data ---
    # prepare_dataset handles reading CSVs, engineering features, and saving to cache
    x_train, y_train = prepare_dataset(
        "train", debug=debug, load_cached_data=load_cached_data
    )
    x_val, y_val = prepare_dataset(
        "val", debug=debug, load_cached_data=load_cached_data
    )
    x_test, _ = prepare_dataset("test", debug=debug, load_cached_data=load_cached_data)

    # --- 3. Robust Scaling ---
    print("Preparing RobustScaler...")

    # Identify features to scale (exclude u_out which is a binary mask)
    feature_indices = Config.get_feature_indices()
    u_out_idx = feature_indices.get("u_out")

    if u_out_idx is None:
        print("Warning: 'u_out' not found in features. Scaling all columns.")
        scale_indices = list(range(x_train.shape[-1]))
    else:
        scale_indices = [i for name, i in feature_indices.items() if name != "u_out"]

    # Flatten tensors to (N * L, F) for sklearn
    N_train, L, F = x_train.shape
    N_val = x_val.shape[0]
    N_test = x_test.shape[0]

    # Convert to numpy for scaling
    x_train_flat = x_train.reshape(-1, F).numpy()
    x_val_flat = x_val.reshape(-1, F).numpy()
    x_test_flat = x_test.reshape(-1, F).numpy()

    scaler = RobustScaler()

    # Fit only on training data, only on continuous features
    print(f"Fitting scaler on {len(scale_indices)} features (excluding u_out)...")
    scaler.fit(x_train_flat[:, scale_indices])

    # Transform all splits
    x_train_flat[:, scale_indices] = scaler.transform(x_train_flat[:, scale_indices])
    x_val_flat[:, scale_indices] = scaler.transform(x_val_flat[:, scale_indices])
    x_test_flat[:, scale_indices] = scaler.transform(x_test_flat[:, scale_indices])

    # Reshape back to tensors
    x_train = torch.tensor(x_train_flat, dtype=torch.float32).reshape(N_train, L, F)
    x_val = torch.tensor(x_val_flat, dtype=torch.float32).reshape(N_val, L, F)
    x_test = torch.tensor(x_test_flat, dtype=torch.float32).reshape(N_test, L, F)

    print("Scaling complete.")

    # --- 4. Create DataLoaders ---
    train_dataset = VentilatorDataset(x_train, y_train, is_test=False)
    val_dataset = VentilatorDataset(x_val, y_val, is_test=False)
    test_dataset = VentilatorDataset(x_test, is_test=True)

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset:   {len(val_dataset)} samples")
    print(f"Test Dataset:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

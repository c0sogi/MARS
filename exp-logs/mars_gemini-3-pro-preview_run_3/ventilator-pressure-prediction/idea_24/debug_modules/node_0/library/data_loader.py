import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    MODEL_FEATURES,
    BATCH_SIZE,
    SEED,
)
from library.features import engineer_features
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, 80, N_features)
            u_out (np.ndarray): Control input u_out of shape (N_breaths, 80)
            y (np.ndarray, optional): Target pressure of shape (N_breaths, 80)
        """
        self.X = torch.FloatTensor(X)
        self.u_out = torch.FloatTensor(u_out)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.u_out[idx]
        else:
            # Return dummy target for test set to maintain consistent signature
            return self.X[idx], torch.zeros_like(self.u_out[idx]), self.u_out[idx]


def get_data_loaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    Handles feature engineering, scaling, and reshaping.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): Whether to load pre-processed numpy arrays from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(SEED)

    # Cache file paths
    cache_files = {
        "train_x": os.path.join(WORKING_DIR, "train_x.npy"),
        "train_y": os.path.join(WORKING_DIR, "train_y.npy"),
        "train_u_out": os.path.join(WORKING_DIR, "train_u_out.npy"),
        "val_x": os.path.join(WORKING_DIR, "val_x.npy"),
        "val_y": os.path.join(WORKING_DIR, "val_y.npy"),
        "val_u_out": os.path.join(WORKING_DIR, "val_u_out.npy"),
        "test_x": os.path.join(WORKING_DIR, "test_x.npy"),
        "test_u_out": os.path.join(WORKING_DIR, "test_u_out.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
        "scaler": os.path.join(WORKING_DIR, "scaler.joblib"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading pre-processed data from cache...")
        train_x = np.load(cache_files["train_x"])
        train_y = np.load(cache_files["train_y"])
        train_u_out = np.load(cache_files["train_u_out"])

        val_x = np.load(cache_files["val_x"])
        val_y = np.load(cache_files["val_y"])
        val_u_out = np.load(cache_files["val_u_out"])

        test_x = np.load(cache_files["test_x"])
        test_u_out = np.load(cache_files["test_u_out"])
        # test_ids are not needed for the loader but good to verify existence

    else:
        print("Processing data from scratch...")

        # 1. Engineer Features
        # engineer_features handles its own caching of the DataFrame
        train_df = engineer_features(TRAIN_PATH, "train_features", load_cached_data)
        val_df = engineer_features(VAL_PATH, "val_features", load_cached_data)
        test_df = engineer_features(TEST_PATH, "test_features", load_cached_data)

        # 2. Prepare Scaler
        print("Fitting RobustScaler...")
        scaler = RobustScaler()

        # Fit only on training data
        # Extract features defined in config
        X_train_raw = train_df[MODEL_FEATURES].values
        scaler.fit(X_train_raw)

        # Save scaler
        joblib.dump(scaler, cache_files["scaler"])

        # 3. Transform and Reshape
        # Constants
        BREATH_STEPS = 80

        def process_split(df, is_test=False):
            # Extract features and transform
            X_raw = df[MODEL_FEATURES].values
            X_scaled = scaler.transform(X_raw)

            # Reshape to (N_breaths, 80, N_features)
            # Ensure the total rows are divisible by 80
            assert len(df) % BREATH_STEPS == 0, "Data length not divisible by 80"
            num_breaths = len(df) // BREATH_STEPS

            X_reshaped = X_scaled.reshape(num_breaths, BREATH_STEPS, -1)

            # Extract u_out for masking (it's also in X, but needed separately for loss)
            u_out_reshaped = df["u_out"].values.reshape(num_breaths, BREATH_STEPS)

            if not is_test:
                y_reshaped = df["pressure"].values.reshape(num_breaths, BREATH_STEPS)
                return X_reshaped, y_reshaped, u_out_reshaped, None
            else:
                # For test, we might need IDs for submission reconstruction later
                # though the loader doesn't strictly need them
                ids_reshaped = df["id"].values.reshape(num_breaths, BREATH_STEPS)
                return X_reshaped, None, u_out_reshaped, ids_reshaped

        print("Transforming and reshaping training data...")
        train_x, train_y, train_u_out, _ = process_split(train_df, is_test=False)

        print("Transforming and reshaping validation data...")
        val_x, val_y, val_u_out, _ = process_split(val_df, is_test=False)

        print("Transforming and reshaping test data...")
        test_x, _, test_u_out, test_ids = process_split(test_df, is_test=True)

        # 4. Save to Cache
        print("Saving processed arrays to cache...")
        np.save(cache_files["train_x"], train_x)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["train_u_out"], train_u_out)

        np.save(cache_files["val_x"], val_x)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["val_u_out"], val_u_out)

        np.save(cache_files["test_x"], test_x)
        np.save(cache_files["test_u_out"], test_u_out)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_u_out, train_y)
    val_dataset = VentilatorDataset(val_x, val_u_out, val_y)
    test_dataset = VentilatorDataset(test_x, test_u_out, y=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

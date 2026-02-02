import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.model import VentilatorDataset, prepare_data, compute_features


def preprocess_dataframe(df):
    """
    Generates PID state features (integral, derivative) and physics interaction terms.
    Wraps the compute_features function from library.model to ensure consistency
    with the model's expected input format.

    Args:
        df (pd.DataFrame): Raw dataframe containing breath time series.

    Returns:
        np.ndarray: Reshaped and engineered features (N_breaths, 80, N_features).
    """
    return compute_features(df)


def get_dataloaders(
    batch_size=Config.HYPERPARAMS["batch_size"],
    num_workers=Config.HYPERPARAMS["num_workers"],
    load_cached_data=True,
):
    """
    Orchestrates data loading, preprocessing, and DataLoader creation.

    This function utilizes the centralized prepare_data pipeline to:
    1. Load raw metadata.
    2. Compute PID and physics features (if not cached).
    3. Scale features using StandardScaler.
    4. Cache processed arrays to disk for fast reloading.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache missing, re-computes features.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Retrieve processed data arrays (Train, Val, Test)
    # prepare_data handles the caching logic, feature engineering, and scaling.
    # It respects Config.FORCE_RECOMPUTE and the load_cached_data flag.
    X_train, y_train, X_val, y_val, X_test = prepare_data(
        load_cached_data=load_cached_data
    )

    # Instantiate Datasets
    # VentilatorDataset converts numpy arrays to torch FloatTensors
    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)
    test_dataset = VentilatorDataset(X_test)

    # Create DataLoaders
    # Pin memory is enabled for faster host-to-device transfer on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.HYPERPARAMS["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.HYPERPARAMS["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.HYPERPARAMS["pin_memory"],
    )

    return train_loader, val_loader, test_loader

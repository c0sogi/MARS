import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config, process_data, CoverTypeDataset, feature_engineering


class FeatureEngineeringPipeline:
    """
    Implements the physics-informed feature engineering pipeline.
    Wraps the logic provided in library.config to generate:
    - Aspect_Sin/Cos (Cyclical)
    - Hydrology_Dist (Geometric Magnitude)
    - Abs_Hydro_Elev (Directional Preservation)
    - Mean_Amenities (Global Context)
    """

    @staticmethod
    def transform(df):
        return feature_engineering(df)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug_size=None,
    num_workers=4,
    pin_memory=True,
):
    """
    Orchestrates data loading, processing, caching, and DataLoader creation.

    Args:
        batch_size (int): Number of samples per batch.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files from cache.
                                 If False or cache miss, re-processes from raw metadata.
        debug_size (int, optional): If provided, limits the dataset size for debugging purposes.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): If True, copies tensors into CUDA pinned memory.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # process_data handles the caching logic (checking idea_33 folder),
    # feature engineering, and standardization.
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=load_cached_data
    )

    # Apply debugging subset if requested
    if debug_size is not None:
        print(f"Debug Mode: Subsetting datasets to {debug_size} samples.")
        X_train = X_train[:debug_size]
        y_train = y_train[:debug_size]
        X_val = X_val[:debug_size]
        y_val = y_val[:debug_size]
        X_test = X_test[:debug_size]
        test_ids = test_ids[:debug_size]

    # Initialize Datasets
    # CoverTypeDataset handles conversion to Float32/Long tensors
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, test_ids

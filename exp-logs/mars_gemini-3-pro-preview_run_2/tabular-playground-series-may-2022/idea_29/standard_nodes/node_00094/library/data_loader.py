import torch
from torch.utils.data import DataLoader
from library.config import Config, ManufacturingDataset, load_and_process_data


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, data_fraction=1.0, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        data_fraction (float): Fraction of data to use (0.0 < fraction <= 1.0).
                               Useful for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data using the centralized logic in library.config
    # This handles reading CSVs, Metadata, Preprocessing, and Caching.
    data = load_and_process_data(load_cached_data=load_cached_data)

    (
        X_train_cont,
        X_train_cat,
        y_train,
        X_val_cont,
        X_val_cat,
        y_val,
        X_test_cont,
        X_test_cat,
        test_ids,
    ) = data

    # Apply data fraction for debugging if requested
    if data_fraction < 1.0:
        n_train = int(len(X_train_cont) * data_fraction)
        n_val = int(len(X_val_cont) * data_fraction)
        n_test = int(len(X_test_cont) * data_fraction)

        X_train_cont = X_train_cont[:n_train]
        X_train_cat = X_train_cat[:n_train]
        y_train = y_train[:n_train]

        X_val_cont = X_val_cont[:n_val]
        X_val_cat = X_val_cat[:n_val]
        y_val = y_val[:n_val]

        X_test_cont = X_test_cont[:n_test]
        X_test_cat = X_test_cat[:n_test]

    # Instantiate Datasets
    # ManufacturingDataset is imported from library.config to avoid duplication
    train_dataset = ManufacturingDataset(X_train_cont, X_train_cat, y_train)
    val_dataset = ManufacturingDataset(X_val_cont, X_val_cat, y_val)
    test_dataset = ManufacturingDataset(X_test_cont, X_test_cat, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_test_ids(load_cached_data=True):
    """
    Helper to retrieve test IDs for submission generation.
    """
    data = load_and_process_data(load_cached_data=load_cached_data)
    # The last element is test_ids
    return data[-1]

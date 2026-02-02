import torch
from torch.utils.data import DataLoader
from library.config import get_data, ManufacturingDataset, BATCH_SIZE, NUM_WORKERS


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns PyTorch DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        num_workers (int): The number of worker processes for data loading.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data using the library function (handles caching and preprocessing)
    data = get_data(load_cached_data=load_cached_data)

    # Unpack the returned tuple
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    ) = data

    # Create Dataset instances
    # ManufacturingDataset is imported from library.config to avoid re-implementation
    train_dataset = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    # Test dataset has no targets
    test_dataset = ManufacturingDataset(X_cat_test, X_cont_test, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def get_test_ids(load_cached_data=True):
    """
    Retrieves the IDs for the test set samples.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        numpy.ndarray: Array of test IDs.
    """
    # get_data returns test_ids as the last element of the tuple
    data = get_data(load_cached_data=load_cached_data)
    return data[-1]

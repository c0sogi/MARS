import torch
from torch.utils.data import DataLoader
from library.config import ManufacturingDataset, load_and_process_data


def get_dataloaders(
    config, load_cached_data=True, max_train_samples=None, max_val_samples=None
):
    """
    Creates DataLoaders for train, validation, and test sets with optional subsetting for debugging.

    Args:
        config: Configuration object containing hyperparameters.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_train_samples (int, optional): Limit training data size for debugging.
        max_val_samples (int, optional): Limit validation data size for debugging.

    Returns:
        tuple: (train_dl, val_dl, test_dl, data_dict)
    """
    # Load processed data using the library function which handles caching,
    # metadata-based splitting, and feature engineering.
    data = load_and_process_data(config, load_cached_data=load_cached_data)

    # Extract arrays from the returned dictionary
    X_seq_train = data["X_seq_train"]
    X_cont_train = data["X_cont_train"]
    y_train = data["y_train"]

    X_seq_val = data["X_seq_val"]
    X_cont_val = data["X_cont_val"]
    y_val = data["y_val"]

    X_seq_test = data["X_seq_test"]
    X_cont_test = data["X_cont_test"]

    # Apply subsetting if requested (for debugging or quick iteration)
    if max_train_samples is not None and max_train_samples < len(X_seq_train):
        X_seq_train = X_seq_train[:max_train_samples]
        X_cont_train = X_cont_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    if max_val_samples is not None and max_val_samples < len(X_seq_val):
        X_seq_val = X_seq_val[:max_val_samples]
        X_cont_val = X_cont_val[:max_val_samples]
        y_val = y_val[:max_val_samples]

    # Instantiate Datasets
    train_ds = ManufacturingDataset(X_seq_train, X_cont_train, y_train)
    val_ds = ManufacturingDataset(X_seq_val, X_cont_val, y_val)
    test_ds = ManufacturingDataset(X_seq_test, X_cont_test, None)

    # Instantiate DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl, data

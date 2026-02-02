import torch
from torch.utils.data import DataLoader
from library import config

# Re-export RNADataset and process_data from config as they are already implemented there.
# This satisfies the requirement to use provided files and avoids re-definition.
RNADataset = config.RNADataset
process_data = config.process_data


def get_loaders(load_cached_data=True, batch_size=None):
    """
    Prepares and returns DataLoaders for the training and validation sets.

    This function handles the loading of processed data (either from cache or by
    processing raw metadata), instantiates the RNADataset, and wraps them in
    PyTorch DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files
                                 from the working directory. If False or file missing,
                                 re-processes data from metadata CSVs.
        batch_size (int, optional): Batch size for the loaders. If None, uses the
                                    value defined in config.BATCH_SIZE.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Load or process data dictionaries
    # process_data handles the caching logic internally (saving/loading .npz)
    # and ensures the working directory exists.
    train_data_dict = process_data(mode="train", load_cached_data=load_cached_data)
    val_data_dict = process_data(mode="val", load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = RNADataset(train_data_dict)
    val_dataset = RNADataset(val_data_dict)

    # Initialize DataLoaders
    # Using multiple workers for efficient data loading given available vCPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True, batch_size=None):
    """
    Prepares and returns a DataLoader for the test set.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.
        batch_size (int, optional): Batch size for the loader.

    Returns:
        DataLoader: The test data loader.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Load or process test data
    test_data_dict = process_data(mode="test", load_cached_data=load_cached_data)

    # Initialize Dataset
    test_dataset = RNADataset(test_data_dict)

    # Initialize DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader

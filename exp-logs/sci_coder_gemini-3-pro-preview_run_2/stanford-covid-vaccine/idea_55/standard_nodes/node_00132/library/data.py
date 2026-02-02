import os
import torch
from torch.utils.data import DataLoader
from library.config import Config, RNADataset, get_data


def get_dataloaders(mode="train", load_cached=True, batch_size=None, num_workers=2):
    """
    Creates and returns a PyTorch DataLoader for the specified dataset split.

    This function delegates the data loading, preprocessing, and caching mechanisms
    to the `get_data` function from `library.config`. It ensures that data is
    loaded efficiently and consistently.

    Args:
        mode (str): The dataset split to load. Options are "train", "val", or "test".
        load_cached (bool): If True, attempts to load pre-processed data from the
                            cache directory defined in Config. If False or if the
                            cache is missing, processes the data from scratch and
                            updates the cache.
        batch_size (int, optional): The number of samples per batch. If None,
                                    defaults to Config.BATCH_SIZE.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader containing the requested dataset.
    """
    # Set default batch size from Config if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Determine shuffling strategy: Shuffle only for the training set
    shuffle = mode == "train"

    # Load data using the centralized get_data function from library.config
    # This handles:
    # 1. Checking for cached .npz files in Config.CACHE_DIR
    # 2. Loading metadata CSVs if cache is missed
    # 3. Processing features (One-Hot, Partner Indices, etc.) via process_dataframe
    # 4. Saving processed data to cache for future use
    X, partners, y, ids = get_data(mode=mode, load_cached=load_cached)

    # Initialize the Dataset
    # RNADataset (imported from library.config) handles tensor conversion
    # and permutes X from (N, L, C) to (N, C, L) for Conv1d compatibility.
    dataset = RNADataset(X, partners, y)

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Optimize memory transfer to GPU if available
    )

    return loader

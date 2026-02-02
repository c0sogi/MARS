import os
import torch
from torch.utils.data import DataLoader
from library.config import Config, RNADataset, get_dataset


def get_dataloader(
    mode: str,
    config: Config = None,
    batch_size: int = None,
    num_workers: int = None,
    shuffle: bool = None,
    load_cached_data: bool = True,
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified mode.

    This function leverages the logic in library.config to:
    1. Load data from Parquet files (or cache).
    2. Process sequences, structures, and targets.
    3. Instantiate the RNADataset.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
        config (Config, optional): Configuration object containing hyperparameters.
                                   Defaults to Config().
        batch_size (int, optional): Batch size. Defaults to config.batch_size.
        num_workers (int, optional): Number of worker processes. Defaults to config.num_workers.
        shuffle (bool, optional): Whether to shuffle the data.
                                  Defaults to True for 'train', False otherwise.
        load_cached_data (bool, optional): Whether to attempt loading from cache. Defaults to True.

    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """
    # Initialize config if not provided
    if config is None:
        config = Config()

    # Set defaults based on config
    if batch_size is None:
        batch_size = config.batch_size

    if num_workers is None:
        num_workers = config.num_workers

    # Default shuffle logic: Shuffle only for training
    if shuffle is None:
        shuffle = mode == "train"

    # Retrieve processed data
    # get_dataset handles:
    # - Loading from Parquet
    # - Tokenization of sequences and loops
    # - Computation of signed pairing distances
    # - Target filtering (3 scored columns + errors)
    # - Caching mechanism (saving/loading .npz files)
    data_dict = get_dataset(mode=mode, load_cached_data=load_cached_data)

    # Instantiate the Dataset
    dataset = RNADataset(data_dict, mode=mode)

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(config.device == "cuda"),
    )

    return loader

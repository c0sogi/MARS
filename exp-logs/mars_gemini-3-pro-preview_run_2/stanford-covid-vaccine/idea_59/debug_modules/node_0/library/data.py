import os
import torch
from torch.utils.data import DataLoader
from library.config import process_data, RNADataset, BATCH_SIZE, set_seed, SEED

# Set fixed random seed for reproducibility
set_seed(SEED)


class RNA_Dataset(RNADataset):
    """
    Dataset class for RNA data.
    Inherits from the provided RNADataset to utilize the existing initialization logic.
    Adds functionality to limit the dataset size for debugging purposes.
    """

    def __init__(self, data, limit=None):
        """
        Args:
            data: Dictionary or NpzFile containing 'inputs', 'targets', 'pairs', 'ids'.
            limit (int, optional): If provided, limits the dataset to the first `limit` samples.
        """
        # Initialize using the parent class logic (converts numpy to torch tensors)
        super().__init__(data)

        # Apply slicing if a limit is specified
        if limit is not None:
            self.inputs = self.inputs[:limit]
            self.targets = self.targets[:limit]
            self.pairs = self.pairs[:limit]
            self.ids = self.ids[:limit]


class Preprocessor:
    """
    Handles data loading, preprocessing, and caching.
    """

    def __init__(self):
        pass

    def process(self, load_cached_data=True):
        """
        Orchestrates the data processing pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from cache.
                                     If False or cache is missing, re-processes raw data.

        Returns:
            dict: A dictionary containing 'train', 'val', and 'test' data objects.
        """
        # Delegate to the provided process_data function which handles:
        # 1. Loading metadata
        # 2. One-Hot encoding (Sequence, Structure, Loop Type)
        # 3. Partner Identity generation
        # 4. Target parsing
        # 5. Caching to disk
        return process_data(load_cached_data=load_cached_data)


def get_loaders(data, batch_size=BATCH_SIZE, debug=False):
    """
    Constructs DataLoaders for the training, validation, and test sets.

    Args:
        data (dict): Dictionary containing processed data (output of Preprocessor.process).
        batch_size (int): Batch size for the DataLoaders.
        debug (bool): If True, limits the dataset size to a small subset for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Determine limit for debugging
    limit = 100 if debug else None

    # Initialize Datasets with optional limit
    train_dataset = RNA_Dataset(data["train"], limit=limit)
    val_dataset = RNA_Dataset(data["val"], limit=limit)
    test_dataset = RNA_Dataset(data["test"], limit=limit)

    # Create DataLoaders
    # num_workers=2 is used as per the provided configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader

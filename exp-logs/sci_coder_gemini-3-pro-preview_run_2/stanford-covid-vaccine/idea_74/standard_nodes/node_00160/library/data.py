import torch
from torch.utils.data import DataLoader
from library.config import Config, process_data, RNADataset

# The RNADataset class and process_data function are imported from library.config
# to strictly adhere to the instruction: "Import the functions or classes from the
# given Python files instead of re-implementing them."
#
# The imported process_data function implements the required caching logic
# (checking ./working/idea_74/ for .npz files) and feature generation
# (One-Hot encoding, Partner Identity, Partner Index Map).
# The imported RNADataset class implements the __getitem__ logic returning
# (inputs, pairs, targets).


def get_loaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Constructs and returns DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): The batch size for the loaders. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache missing, re-processes data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Retrieve processed data.
    # This function handles the logic of checking cache vs computing from scratch.
    data_map = process_data(load_cached_data=load_cached_data)

    # Instantiate Datasets using the data dictionaries
    train_dataset = RNADataset(data_map["train"], mode="train")
    val_dataset = RNADataset(data_map["val"], mode="val")
    test_dataset = RNADataset(data_map["test"], mode="test")

    # Instantiate DataLoaders
    # Use multiple workers and pin_memory for efficient data transfer to GPU
    num_workers = 2
    pin_memory = torch.cuda.is_available()

    # Train loader: Shuffle is essential for stochastic gradient descent
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Drop incomplete batches to maintain consistent dimensions
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Val loader: Sequential access for deterministic evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Test loader: Sequential access for inference
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

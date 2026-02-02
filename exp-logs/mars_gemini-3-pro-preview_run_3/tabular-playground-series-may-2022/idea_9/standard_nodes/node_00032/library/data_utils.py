import torch
from torch.utils.data import DataLoader
from library.config import Config, TabularDataset, process_data


class ManufacturingDataset(TabularDataset):
    """
    Dataset class for manufacturing control data.
    Inherits from TabularDataset provided in library.config to utilize
    the existing implementation for handling continuous and categorical data.
    """

    pass


def get_data_loaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_limit=None
):
    """
    Executes the feature engineering pipeline and returns DataLoaders for training, validation, and testing.

    The pipeline (delegated to library.config.process_data) includes:
    1. Loading data from metadata CSVs.
    2. Splitting f_27 into 10 character columns.
    3. Computing unique_character_count.
    4. Transductive Ordinal Encoding (fitted on Train+Val+Test).
    5. StandardScaler normalization for continuous features.
    6. Caching of processed tensors.

    Args:
        batch_size (int): The batch size for the DataLoaders. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): Whether to attempt loading processed data from the cache.
        debug_limit (int, optional): If provided, limits the size of the training set for debugging purposes.

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab_sizes, cont_dim)
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - vocab_sizes (list): List of integers representing the vocabulary size for each categorical feature.
            - cont_dim (int): The number of continuous features.
    """
    # Execute the feature engineering pipeline using the provided library function
    data = process_data(load_cached_data=load_cached_data)

    train_cont = data["train_cont"]
    train_cat = data["train_cat"]
    train_y = data["train_y"]

    val_cont = data["val_cont"]
    val_cat = data["val_cat"]
    val_y = data["val_y"]

    test_cont = data["test_cont"]
    test_cat = data["test_cat"]

    # Apply debug limit to training data if specified
    if debug_limit is not None and debug_limit < len(train_cont):
        print(f"Debugging: Limiting training data to {debug_limit} samples.")
        train_cont = train_cont[:debug_limit]
        train_cat = train_cat[:debug_limit]
        train_y = train_y[:debug_limit]

    # Instantiate Datasets
    train_ds = ManufacturingDataset(train_cont, train_cat, train_y)
    val_ds = ManufacturingDataset(val_cont, val_cat, val_y)
    test_ds = ManufacturingDataset(test_cont, test_cat)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, data["vocab_sizes"], data["cont_dim"]

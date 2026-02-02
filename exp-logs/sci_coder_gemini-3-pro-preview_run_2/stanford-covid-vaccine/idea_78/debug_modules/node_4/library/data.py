import torch
from torch.utils.data import DataLoader
from library.config import Config, RNADataset, load_or_process_data


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, train_shuffle=True
):
    """
    Creates and returns DataLoaders for training, validation, and testing.

    This function acts as a high-level interface to the data processing logic
    defined in library.config. It ensures that data is loaded from the cache
    if available, or processed from the metadata files if not.

    Args:
        batch_size (int): The number of samples per batch. Defaults to Config.BATCH_SIZE.
        num_workers (int): The number of subprocesses to use for data loading.
                           Defaults to Config.NUM_WORKERS.
        train_shuffle (bool): Whether to shuffle the training dataset. Defaults to True.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
            - train_loader (DataLoader): Loader for the training set.
            - val_loader (DataLoader): Loader for the validation set.
            - test_loader (DataLoader): Loader for the test set.
            - test_ids (np.ndarray): Array of sample IDs for the test set (used for submission).
    """

    # --------------------------------------------------------------------------
    # 1. Training Data
    # --------------------------------------------------------------------------
    # load_or_process_data handles:
    # - Loading from .npz cache if it exists
    # - Reading from metadata/train.csv if cache is missing
    # - Generating features (One-Hot, Partner Identity)
    # - Saving to cache
    train_ids, train_X, train_P, train_Y = load_or_process_data("train")

    train_dataset = RNADataset(train_X, train_P, train_Y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------------------
    # 2. Validation Data
    # --------------------------------------------------------------------------
    val_ids, val_X, val_P, val_Y = load_or_process_data("val")

    val_dataset = RNADataset(val_X, val_P, val_Y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------------------
    # 3. Test Data
    # --------------------------------------------------------------------------
    # Note: Test data processing returns 3 values (no targets)
    test_ids, test_X, test_P = load_or_process_data("test")

    test_dataset = RNADataset(test_X, test_P)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, test_ids

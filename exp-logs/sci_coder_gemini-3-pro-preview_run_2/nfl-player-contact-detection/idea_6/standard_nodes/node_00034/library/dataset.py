import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processing import prepare_data


class ContactDataset(Dataset):
    """
    PyTorch Dataset wrapper for the NFL Contact Detection task.
    Simplified for K-MLP.
    """

    def __init__(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): The X_wide tensor.
            targets (torch.Tensor or np.array): Target labels (for train/val) or contact_ids (for test).
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(
    load_cached_data=True,
    debug=False,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to load pre-processed features from disk.
        debug (bool): If True, uses a small subset of data for debugging.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Retrieve processed tensors from the data processing library
    # prepare_data handles caching and raw data loading internally
    train_data, val_data, test_data = prepare_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Instantiate Datasets
    # train_data is (X_wide, targets)
    train_dataset = ContactDataset(train_data[0], train_data[1])
    val_dataset = ContactDataset(val_data[0], val_data[1])
    test_dataset = ContactDataset(test_data[0], test_data[1])

    # 3. Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

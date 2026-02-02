import os
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import get_data


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps pre-processed tensors for inputs, targets, and masks.
    """

    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing tensors 'x', 'u_out', and optionally 'y' or 'ids'.
            is_test (bool): Flag indicating if this is the test set (returns ids instead of y).
        """
        self.x = data_dict["x"]
        self.u_out = data_dict["u_out"]
        self.is_test = is_test

        if self.is_test:
            self.ids = data_dict["ids"]
            self.y = None
        else:
            self.y = data_dict["y"]
            self.ids = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a tuple of tensors for the given index.

        Train/Val: (x, y, u_out)
        Test:      (x, u_out, ids)
        """
        x = self.x[idx]
        u_out = self.u_out[idx]

        if self.is_test:
            ids = self.ids[idx]
            return x, u_out, ids
        else:
            y = self.y[idx]
            return x, y, u_out


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading processed data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure the required directory exists as per specific task instructions
    os.makedirs("./working/idea_7", exist_ok=True)

    # Delegate data loading and processing to the provided library
    # This handles caching, feature engineering, scaling, and reshaping
    train_data, val_data, test_data = get_data(load_cached_data=load_cached_data)

    # Create Dataset objects
    train_dataset = VentilatorDataset(train_data, is_test=False)
    val_dataset = VentilatorDataset(val_data, is_test=False)
    test_dataset = VentilatorDataset(test_data, is_test=True)

    # Create DataLoaders
    # Pin memory is beneficial for GPU training
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

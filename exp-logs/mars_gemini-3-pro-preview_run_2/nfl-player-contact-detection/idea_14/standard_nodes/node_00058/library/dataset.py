import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.data_processing import DataProcessor


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the NFL Contact Detection task.
    Wraps preprocessed continuous and categorical features into Tensors.
    """

    def __init__(self, X_cont, X_cat, y=None):
        """
        Args:
            X_cont (np.ndarray): Continuous features matrix. Shape (N, num_continuous).
            X_cat (np.ndarray): Categorical features matrix. Shape (N, num_categorical).
            y (np.ndarray, optional): Target labels. Shape (N, ) or (N, 1).
        """
        # Convert to tensors immediately.
        # The dataset fits in memory (220GB RAM available), so this avoids overhead during __getitem__.
        self.X_cont = torch.as_tensor(X_cont, dtype=torch.float32)
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)

        if y is not None:
            # Ensure y is float32 and has shape (N, 1) for BCEWithLogitsLoss compatibility
            # FocalLoss expects targets to match input shape (Batch, 1)
            self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (x_cont, x_cat, y) if labels exist, else (x_cont, x_cat)
        """
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_cont[idx], self.X_cat[idx]


def get_dataloaders(config: Config, processor: DataProcessor):
    """
    Loads training and validation data using the processor and returns DataLoaders.

    Args:
        config (Config): Configuration object containing batch size and worker settings.
        processor (DataProcessor): Instance of DataProcessor to handle data loading/caching.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load and process data (uses caching internally via DataProcessor)
    # This returns numpy arrays
    X_train, X_cat_train, y_train, X_val, X_cat_val, y_val = (
        processor.load_and_process_train_val(load_cached_data=True)
    )

    # Wrap in Dataset class
    train_dataset = ContactDataset(X_train, X_cat_train, y_train)
    val_dataset = ContactDataset(X_val, X_cat_val, y_val)

    # Configure hardware acceleration settings
    use_pin_memory = config.DEVICE == "cuda"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batches to ensure stable gradients (esp. with LayerNorm)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(config: Config, processor: DataProcessor):
    """
    Loads test data using the processor and returns the DataLoader and contact IDs.

    Args:
        config (Config): Configuration object.
        processor (DataProcessor): Instance of DataProcessor.

    Returns:
        tuple: (test_loader, test_ids)
    """
    # Load and process test data
    X_test, X_cat_test, test_ids = processor.load_and_process_test(
        load_cached_data=True
    )

    # Wrap in Dataset class (no labels provided)
    test_dataset = ContactDataset(X_test, X_cat_test, y=None)

    use_pin_memory = config.DEVICE == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return test_loader, test_ids

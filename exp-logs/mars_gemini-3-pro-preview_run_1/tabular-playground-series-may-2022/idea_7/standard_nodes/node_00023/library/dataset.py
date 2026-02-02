import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles numerical features, tokenized sequence features, and binary targets.
    """

    def __init__(self, X_num, X_seq, y=None):
        """
        Args:
            X_num (np.ndarray): Normalized numerical features.
            X_seq (np.ndarray): Tokenized sequence features.
            y (np.ndarray, optional): Target labels.
        """
        # Convert to tensors
        # Numerical features -> Float32
        self.X_num = torch.tensor(X_num, dtype=torch.float32)

        # Sequence features -> Long (int64) for Embedding layers
        self.X_seq = torch.tensor(X_seq, dtype=torch.long)

        # Targets -> Float32 for BCEWithLogitsLoss (requires shape [N, 1])
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        item = {"numerical": self.X_num[idx], "sequence": self.X_seq[idx]}

        if self.y is not None:
            item["target"] = self.y[idx]

        return item


def create_dataloaders(
    data_dict,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    pin_memory=Config.PIN_MEMORY,
    limit_samples=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        data_dict (dict): Dictionary containing processed numpy arrays (from preprocess_pipeline).
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        limit_samples (int, optional): If provided, limits the dataset size for debugging purposes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # helper to slice data if limit_samples is set
    def get_subset(arr, limit):
        if limit is not None and limit < len(arr):
            return arr[:limit]
        return arr

    # --- Training Data ---
    X_num_train = get_subset(data_dict["X_num_train"], limit_samples)
    X_seq_train = get_subset(data_dict["X_seq_train"], limit_samples)
    y_train = get_subset(data_dict["y_train"], limit_samples)

    train_ds = ManufacturingDataset(X_num_train, X_seq_train, y_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain consistent statistics
    )

    # --- Validation Data ---
    X_num_val = get_subset(data_dict["X_num_val"], limit_samples)
    X_seq_val = get_subset(data_dict["X_seq_val"], limit_samples)
    y_val = get_subset(data_dict["y_val"], limit_samples)

    val_ds = ManufacturingDataset(X_num_val, X_seq_val, y_val)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # --- Test Data ---
    X_num_test = get_subset(data_dict["X_num_test"], limit_samples)
    X_seq_test = get_subset(data_dict["X_seq_test"], limit_samples)
    # Test set has no targets

    test_ds = ManufacturingDataset(X_num_test, X_seq_test, y=None)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

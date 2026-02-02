import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.data_processing import create_datasets


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the RVC-Net.
    Handles dual-stream inputs (Kinematic and Visual) and binary labels.
    """

    def __init__(self, X_kin, X_vis, y=None):
        """
        Args:
            X_kin (np.ndarray): Kinematic features matrix.
            X_vis (np.ndarray): Visual features matrix.
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # Convert inputs to torch tensors
        self.X_kin = torch.tensor(X_kin, dtype=torch.float32)
        self.X_vis = torch.tensor(X_vis, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: ((kinematic_features, visual_features), label)
        """
        k_feat = self.X_kin[idx]
        v_feat = self.X_vis[idx]

        # If labels exist, return them; otherwise return a dummy placeholder
        if self.y is not None:
            label = self.y[idx]
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        return (k_feat, v_feat), label


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Factory function to create DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from disk.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker threads. Defaults to Config.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use defaults from Config if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load raw numpy arrays using the provided data processing library
    # The create_datasets function handles the caching logic internally
    train_data, val_data, test_data = create_datasets(load_cached_data=load_cached_data)

    # Unpack tuples returned by create_datasets
    # Structure: (X_kin, X_vis, y, ids) for train/val
    # Structure: (X_kin, X_vis, ids, df) for test
    train_X_kin, train_X_vis, train_y, _ = train_data
    val_X_kin, val_X_vis, val_y, _ = val_data
    test_X_kin, test_X_vis, _, _ = test_data

    # Instantiate Datasets
    train_dataset = ContactDataset(train_X_kin, train_X_vis, train_y)
    val_dataset = ContactDataset(val_X_kin, val_X_vis, val_y)
    test_dataset = ContactDataset(test_X_kin, test_X_vis, y=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop incomplete batch to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader

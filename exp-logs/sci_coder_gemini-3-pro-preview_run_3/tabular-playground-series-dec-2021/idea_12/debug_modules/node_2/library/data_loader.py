import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import FeatureEngineer


class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type prediction task.
    Handles efficient conversion to tensors and supports both training (features + targets)
    and testing (features + ids) modes.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels.
            ids (np.ndarray, optional): Sample IDs (for test set).
        """
        # Convert to tensors immediately to avoid overhead in __getitem__
        # Features are float32, Targets are long (int64) for CrossEntropyLoss
        self.X = torch.from_numpy(X).float()

        self.y = None
        if y is not None:
            self.y = torch.from_numpy(y).long()

        self.ids = None
        if ids is not None:
            self.ids = torch.from_numpy(ids).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (features, target) if y is present.
            tuple: (features, id) if ids are present (and no y).
            tensor: features if neither y nor ids are present.
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        elif self.ids is not None:
            return self.X[idx], self.ids[idx]
        else:
            return self.X[idx]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Orchestrates the loading, engineering, and wrapping of data into PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for training and inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading pre-processed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Retrieve processed data using the FeatureEngineer library
    engineer = FeatureEngineer()
    train_X, train_y, val_X, val_y, test_X, test_ids = engineer.process_data(
        load_cached_data=load_cached_data
    )

    # 2. Handle Debug Mode (Subsampling)
    if Config.DEBUG:
        print(f"DEBUG mode active: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
        limit = Config.DEBUG_SAMPLE_SIZE

        # Slice training data
        train_X = train_X[:limit]
        train_y = train_y[:limit]

        # Slice validation data
        val_X = val_X[:limit]
        val_y = val_y[:limit]

        # Slice test data
        test_X = test_X[:limit]
        test_ids = test_ids[:limit]

    # 3. Create Dataset Objects
    train_dataset = ForestDataset(train_X, y=train_y)
    val_dataset = ForestDataset(val_X, y=val_y)
    # For test dataset, we pass IDs so the loader returns (features, id) tuples
    test_dataset = ForestDataset(test_X, ids=test_ids)

    # 4. Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer for CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"DataLoaders created: Train batches={len(train_loader)}, "
        f"Val batches={len(val_loader)}, Test batches={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader

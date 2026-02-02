import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config, process_data


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    Wraps pre-processed numpy arrays.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates the loading, processing, and batching of data.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache via library.config.process_data.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
        test_loader (DataLoader)
        num_features (int): Input dimension for the model.
        num_classes (int): Number of target classes.
        test_ids (np.array): IDs corresponding to the test set for submission.
    """
    # 1. Get processed numpy arrays from the config library
    # This handles feature engineering, scaling, and caching internally
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        test_ids,
        num_features,
        num_classes,
        _,  # label_encoder is not needed here
    ) = process_data(load_cached_data=load_cached_data)

    # 2. Create PyTorch Datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y=None)

    # 3. Create DataLoaders
    # Train loader needs shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Val and Test loaders do not need shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, num_features, num_classes, test_ids

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from library.config import Config
from library.features import FeatureGenerator


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Reshapes flattened feature vectors into (Channels, Time) format for 1D-CNNs.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (pd.DataFrame): Flattened feature DataFrame of shape (N, Features * Time).
            y (np.array, optional): Target labels. Defaults to None.
            ids (np.array, optional): Contact IDs. Defaults to None.
        """
        self.ids = ids

        # Dimensions from Config
        self.num_features = Config.NUM_FEATURES
        self.window_size = Config.WINDOW_SIZE

        # 1. Process Features
        # Input X is (N, T*C) where columns are ordered as [t0_c0...t0_cC, t1_c0...t1_cC, ...]
        # We need to reshape this to (N, C, T) for Conv1d input.

        # Convert DataFrame to numpy float32 array
        X_values = X.values.astype(np.float32)

        # Reshape to (N, Time, Features)
        # The data was concatenated along the time axis, so the fast axis is Features.
        try:
            X_reshaped = X_values.reshape(-1, self.window_size, self.num_features)
        except ValueError as e:
            raise ValueError(
                f"Shape mismatch: Expected {self.window_size}*{self.num_features} columns, got {X_values.shape[1]}. Error: {e}"
            )

        # Transpose to (N, Features, Time) -> (N, C, L)
        X_transposed = X_reshaped.transpose(0, 2, 1)

        # Convert to PyTorch Tensor
        self.X = torch.from_numpy(X_transposed)

        # 2. Process Labels
        if y is not None:
            # Ensure y is float32 and shape (N, 1) for BCEWithLogitsLoss
            self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        else:
            # Create dummy labels for inference (Test set)
            self.y = torch.zeros((len(X), 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (feature_tensor, label_tensor)
            feature_tensor shape: (C, T)
            label_tensor shape: (1,)
        """
        return self.X[idx], self.y[idx]


def get_dataloader(
    split, batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=4
):
    """
    Factory function to create DataLoaders for train, validation, and test splits.

    Args:
        split (str): One of 'train', 'validation', 'test'.
        batch_size (int): Batch size for the dataloader.
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """
    # Initialize FeatureGenerator
    fg = FeatureGenerator()

    # Load or Generate Data
    # The FeatureGenerator handles the caching logic internally (checking .parquet/.npy files)
    X, y, ids = fg.process_split(split, load_cached_data=load_cached_data)

    # Initialize Dataset
    dataset = ContactDataset(X, y, ids)

    # Configure DataLoader
    # Shuffle only for the training set
    shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Optimization for GPU training
        drop_last=(split == "train"),  # Drop incomplete batch only during training
    )

    return loader

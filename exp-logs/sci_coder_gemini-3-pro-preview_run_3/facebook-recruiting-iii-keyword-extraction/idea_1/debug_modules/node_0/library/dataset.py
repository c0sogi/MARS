import torch
import scipy.sparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import BATCH_SIZE, NUM_WORKERS, set_seed
from library.data_processor import process_data


class StackExchangeDataset(Dataset):
    """
    Custom PyTorch Dataset for sparse Stack Exchange data.
    Converts sparse SciPy matrices to dense PyTorch tensors on-the-fly to manage memory.
    """

    def __init__(self, features, labels=None):
        """
        Args:
            features (scipy.sparse.spmatrix): Sparse feature matrix (N_samples, N_features).
            labels (scipy.sparse.spmatrix, optional): Sparse label matrix (N_samples, N_labels).
        """
        # Ensure CSR format for efficient row slicing
        self.features = (
            features.tocsr() if not scipy.sparse.isspmatrix_csr(features) else features
        )

        if labels is not None:
            self.labels = (
                labels.tocsr() if not scipy.sparse.isspmatrix_csr(labels) else labels
            )
        else:
            self.labels = None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Extract feature vector
        # .toarray() returns (1, n_features), squeeze to (n_features,)
        x_data = self.features[idx].toarray()
        x = torch.from_numpy(x_data).float().squeeze(0)

        if self.labels is not None:
            # Extract label vector
            y_data = self.labels[idx].toarray()
            y = torch.from_numpy(y_data).float().squeeze(0)
            return x, y
        else:
            return x


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Initializes datasets and dataloaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_engineer)
    """
    set_seed()

    # Load processed data using the library function
    # Unpack the 6 return values: X_train, y_train, X_val, y_val, X_test, fe
    X_train, y_train, X_val, y_val, X_test, fe = process_data(
        load_cached_data=load_cached_data
    )

    # Initialize Datasets
    train_dataset = StackExchangeDataset(X_train, y_train)
    val_dataset = StackExchangeDataset(X_val, y_val)
    test_dataset = StackExchangeDataset(X_test, labels=None)

    # Initialize DataLoaders
    # pin_memory=True enables faster data transfer to CUDA-enabled GPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, fe

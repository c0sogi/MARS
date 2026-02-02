import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.preprocessing import Preprocessor


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Wide-and-Deep model.
    Handles sparse TF-IDF features (Wide), dense integer sequences (Deep),
    and sparse binary targets.
    """

    def __init__(self, X_wide, X_deep, y=None):
        """
        Args:
            X_wide (scipy.sparse.csr_matrix): Sparse TF-IDF features (N, 100000).
            X_deep (np.ndarray): Dense integer sequences (N, 150).
            y (scipy.sparse.csr_matrix, optional): Sparse binary tags (N, 5000).
        """
        # Ensure sparse matrices are in CSR format for efficient row slicing
        self.X_wide = X_wide.tocsr() if sparse.issparse(X_wide) else X_wide
        self.X_deep = X_deep

        self.y = None
        if y is not None:
            self.y = y.tocsr() if sparse.issparse(y) else y

    def __len__(self):
        return self.X_deep.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a single sample.
        Converts sparse rows to dense tensors on the fly.
        """
        # 1. Wide Component (Sparse -> Dense -> Tensor)
        # Slicing a CSR matrix returns a 2D matrix (1, n_features).
        # .toarray() converts to dense numpy, .squeeze(0) flattens to 1D.
        x_wide_data = self.X_wide[idx].toarray().squeeze(0)
        x_wide_tensor = torch.tensor(x_wide_data, dtype=torch.float32)

        # 2. Deep Component (Dense -> Tensor)
        # Already dense numpy array. Convert to LongTensor for embedding lookup.
        x_deep_data = self.X_deep[idx]
        x_deep_tensor = torch.tensor(x_deep_data, dtype=torch.long)

        sample = {"wide": x_wide_tensor, "deep": x_deep_tensor}

        # 3. Target (Sparse -> Dense -> Tensor)
        if self.y is not None:
            y_data = self.y[idx].toarray().squeeze(0)
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
            sample["target"] = y_tensor

        return sample


def get_dataloader(
    split, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create a DataLoader for a specific data split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Initialize Preprocessor
    preprocessor = Preprocessor()

    # Load data using the preprocessor (handles caching logic internally)
    # load_cached_data=True is the default in Preprocessor.load_data,
    # ensuring we use existing files if available.
    X_wide, X_deep, y = preprocessor.load_data(split=split, load_cached_data=True)

    # Instantiate Dataset
    dataset = StackExchangeDataset(X_wide, X_deep, y)

    # Create DataLoader
    # pin_memory=True speeds up transfer to GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader

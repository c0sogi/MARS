import torch
from torch.utils.data import Dataset
import numpy as np
from library.preprocessing import process_data


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    """

    def __init__(self, X_num, X_seq, y=None, ids=None):
        """
        Args:
            X_num (np.ndarray): Normalized numerical features. Shape (N, num_features).
            X_seq (np.ndarray): Tokenized sequences. Shape (N, seq_len).
            y (np.ndarray, optional): Target labels. Shape (N,).
            ids (np.ndarray, optional): Sample IDs. Shape (N,).
        """
        self.X_num = X_num
        self.X_seq = X_seq
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        # Convert inputs to tensors
        # Numerical features -> Float32
        # Sequence indices -> Long (int64) for embedding layers
        item = {
            "numerical": torch.tensor(self.X_num[idx], dtype=torch.float32),
            "sequence": torch.tensor(self.X_seq[idx], dtype=torch.long),
        }

        # Add target if available (Float32 for BCEWithLogitsLoss)
        if self.y is not None:
            item["label"] = torch.tensor(self.y[idx], dtype=torch.float32)

        # Add ID if available (useful for test set submission)
        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def get_datasets(load_cached_data=True, sample_size=None):
    """
    Loads data using the preprocessing library and returns formatted PyTorch Datasets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        sample_size (int, optional): Number of samples to use for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load processed numpy arrays
    data = process_data(load_cached_data=load_cached_data, sample_size=sample_size)

    # Unpack the tuple returned by process_data
    (
        X_train_num,
        X_train_seq,
        y_train,
        X_val_num,
        X_val_seq,
        y_val,
        X_test_num,
        X_test_seq,
        ids_test,
    ) = data

    # Create Dataset objects
    train_ds = ManufacturingDataset(X_train_num, X_train_seq, y_train)
    val_ds = ManufacturingDataset(X_val_num, X_val_seq, y_val)
    # Test dataset includes IDs for submission mapping
    test_ds = ManufacturingDataset(X_test_num, X_test_seq, ids=ids_test)

    return train_ds, val_ds, test_ds

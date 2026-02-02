import torch
from torch.utils.data import Dataset
import numpy as np
from library.data_utils import get_data


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Holds continuous and categorical features in memory as tensors.
    """

    def __init__(self, X_cat, X_cont, y=None):
        """
        Args:
            X_cat (np.ndarray): Categorical features (int64).
            X_cont (np.ndarray): Continuous features (float32).
            y (np.ndarray, optional): Target labels (float32). Defaults to None.
        """
        # Convert to tensors immediately for efficiency
        self.X_cat = torch.from_numpy(X_cat).long()
        self.X_cont = torch.from_numpy(X_cont).float()

        if y is not None:
            # Ensure target is (N, 1) for BCEWithLogitsLoss
            self.y = torch.from_numpy(y).float().unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        item = {"cat": self.X_cat[idx], "cont": self.X_cont[idx]}

        if self.y is not None:
            item["target"] = self.y[idx]

        return item


def get_datasets(load_cached_data=True):
    """
    Loads processed data using library.data_utils and creates PyTorch Datasets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, metadata)
    """
    # Delegate data loading and processing to the utility library
    data = get_data(load_cached_data=load_cached_data)

    # Instantiate datasets
    train_ds = ManufacturingDataset(
        data["train"]["X_cat"], data["train"]["X_cont"], data["train"]["y"]
    )

    val_ds = ManufacturingDataset(
        data["val"]["X_cat"], data["val"]["X_cont"], data["val"]["y"]
    )

    # Test set does not have targets
    test_ds = ManufacturingDataset(
        data["test"]["X_cat"], data["test"]["X_cont"], y=None
    )

    return train_ds, val_ds, test_ds, data["meta"]

import torch
from torch.utils.data import Dataset
import numpy as np
from library import data_utils


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    """

    def __init__(self, continuous_data, categorical_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Standardized numerical features.
            categorical_data (np.ndarray): Integer-encoded categorical tokens.
            targets (np.ndarray, optional): Binary target labels.
        """
        # Convert numpy arrays to PyTorch tensors.
        # Continuous data is float32 for neural network inputs.
        self.continuous = torch.tensor(continuous_data, dtype=torch.float32)

        # Categorical data is converted to long (int64) for Embedding layers.
        self.categorical = torch.tensor(categorical_data, dtype=torch.long)

        # Targets are float32 for BCEWithLogitsLoss.
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        sample = {
            "continuous": self.continuous[idx],
            "categorical": self.categorical[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        return sample


def get_datasets(load_cached_data=True):
    """
    Loads processed data using the data_utils library and creates Dataset instances.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Retrieve dictionary of numpy arrays from data_utils
    data = data_utils.process_data(load_cached_data=load_cached_data)

    # Instantiate Training Dataset
    train_dataset = ManufacturingDataset(
        continuous_data=data["X_num_train"],
        categorical_data=data["X_cat_train"],
        targets=data["y_train"],
    )

    # Instantiate Validation Dataset
    val_dataset = ManufacturingDataset(
        continuous_data=data["X_num_val"],
        categorical_data=data["X_cat_val"],
        targets=data["y_val"],
    )

    # Instantiate Test Dataset (Targets are None)
    test_dataset = ManufacturingDataset(
        continuous_data=data["X_num_test"],
        categorical_data=data["X_cat_test"],
        targets=None,
    )

    return train_dataset, val_dataset, test_dataset

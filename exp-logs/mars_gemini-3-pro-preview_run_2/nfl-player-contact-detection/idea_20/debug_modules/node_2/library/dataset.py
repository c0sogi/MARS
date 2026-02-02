import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library import config, data_processing


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Contact Detection Task.
    Separates inputs into Kinematic and Visual streams.
    """

    def __init__(self, x_kin, x_vis, y=None):
        """
        Args:
            x_kin (np.ndarray): Kinematic features array.
            x_vis (np.ndarray): Visual features array.
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # Convert to FloatTensor.
        # Data is kept on CPU until the DataLoader moves a batch to GPU.
        self.x_kin = torch.FloatTensor(x_kin)
        self.x_vis = torch.FloatTensor(x_vis)

        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_kin)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (x_kinematic, x_visual, target)
            If target is None, returns -1 as placeholder.
        """
        x_k = self.x_kin[idx]
        x_v = self.x_vis[idx]

        if self.y is not None:
            target = self.y[idx]
        else:
            target = torch.tensor(-1.0)  # Placeholder for inference if needed

        return x_k, x_v, target


def get_train_val_loaders(batch_size=None, num_workers=None):
    """
    Retrieves training and validation data and returns DataLoaders.

    Args:
        batch_size (int, optional): Batch size. Defaults to config.TRAIN_PARAMS['batch_size'].
        num_workers (int, optional): Number of workers. Defaults to config.TRAIN_PARAMS['num_workers'].

    Returns:
        tuple: (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.TRAIN_PARAMS["batch_size"]
    if num_workers is None:
        num_workers = config.TRAIN_PARAMS["num_workers"]

    # Load processed data
    # caching is handled inside data_processing functions
    X_kin_train, X_vis_train, y_train, _ = data_processing.get_train_data(
        load_cached_data=True
    )
    X_kin_val, X_vis_val, y_val, _ = data_processing.get_val_data(load_cached_data=True)

    # Instantiate Datasets
    train_dataset = ContactDataset(X_kin_train, X_vis_train, y_train)
    val_dataset = ContactDataset(X_kin_val, X_vis_val, y_val)

    # Create DataLoaders
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

    return train_loader, val_loader


def get_test_loader(batch_size=None, num_workers=None):
    """
    Retrieves test data and returns DataLoader and IDs.

    Args:
        batch_size (int, optional): Batch size. Defaults to config.TRAIN_PARAMS['batch_size'].
        num_workers (int, optional): Number of workers. Defaults to config.TRAIN_PARAMS['num_workers'].

    Returns:
        tuple: (test_loader, test_ids)
    """
    if batch_size is None:
        batch_size = config.TRAIN_PARAMS["batch_size"]
    if num_workers is None:
        num_workers = config.TRAIN_PARAMS["num_workers"]

    # Load processed test data
    X_kin_test, X_vis_test, y_test, ids_test = data_processing.get_test_data(
        load_cached_data=True
    )

    # Instantiate Dataset
    # y_test contains placeholders (0) from sample_submission
    test_dataset = ContactDataset(X_kin_test, X_vis_test, y_test)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, ids_test

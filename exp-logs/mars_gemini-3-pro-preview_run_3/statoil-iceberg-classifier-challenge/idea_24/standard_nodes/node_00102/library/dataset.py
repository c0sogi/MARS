import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from library.utils import load_data


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg vs Ship classification.
    Wraps pre-processed numpy arrays and handles tensor conversion.
    """

    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (N, 3, 75, 75), so self.X[idx] is (3, 75, 75)
        img = torch.from_numpy(self.X[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            # Label needs to be float32 for BCEWithLogitsLoss
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


def get_dataloaders(
    batch_size=32,
    cache_dir="./working/idea_24",
    load_cached_data=True,
    num_workers=2,
    debug=False,
):
    """
    Prepares DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Number of samples per batch.
        cache_dir (str): Directory to store/load processed numpy files.
        load_cached_data (bool): If True, attempts to load from cache.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, truncates datasets for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, ids_test)
    """
    # Load processed data from utils
    data = load_data(cache_dir=cache_dir, load_cached_data=load_cached_data)
    (
        X_train,
        y_train,
        angles_train,
        X_val,
        y_val,
        angles_val,
        X_test,
        ids_test,
        angles_test,
    ) = data

    # Handle debug mode
    if debug:
        print("Debug mode enabled: Truncating datasets.")
        subset_size = 128
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        angles_train = angles_train[:subset_size]

        X_val = X_val[:subset_size]
        y_val = y_val[:subset_size]
        angles_val = angles_val[:subset_size]

        X_test = X_test[:subset_size]
        ids_test = ids_test[:subset_size]
        angles_test = angles_test[:subset_size]

    # Define Augmentations
    # Images are already (C, H, W) tensors when passed to transform
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Instantiate Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)
    test_dataset = IcebergDataset(X_test, angles_test, transform=None)

    # Instantiate DataLoaders
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

    return train_loader, val_loader, test_loader, ids_test

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import load_data


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel radar images and incidence angle metadata.
    """

    def __init__(self, X, meta, y=None, transform=False):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 75, 75, 3).
            meta (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,). Defaults to None.
            transform (bool): Whether to apply data augmentation. Defaults to False.
        """
        self.X = X
        self.meta = meta
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # (75, 75, 3)
        angle = self.meta[idx]

        # Convert to Tensor and rearrange to (C, H, W)
        # Input is (H, W, C), PyTorch expects (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Prepare metadata tensor (Batch, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)

        # Apply Augmentations (Train only)
        if self.transform:
            # 1. Discrete Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])

            # 2. Horizontal Flip
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])

            # Note: Vertical Flip is explicitly excluded per instructions.

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
            return (img_tensor, angle_tensor), label
        else:
            return (img_tensor, angle_tensor)


def get_kfold_loaders(data_dict, batch_size=32, n_splits=5, seed=42):
    """
    Generates DataLoaders for Stratified 5-Fold Cross-Validation.
    Combines the pre-split train/val data from utils.load_data to perform
    a full CV on the available training data.

    Args:
        data_dict (dict): Dictionary containing 'X_train', 'y_train', 'meta_train', etc.
        batch_size (int): Batch size for loaders.
        n_splits (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of tuples (train_loader, val_loader) for each fold.
    """
    # Combine the static split provided by load_data to perform full CV
    X_full = np.concatenate([data_dict["X_train"], data_dict["X_val"]], axis=0)
    y_full = np.concatenate([data_dict["y_train"], data_dict["y_val"]], axis=0)
    meta_full = np.concatenate([data_dict["meta_train"], data_dict["meta_val"]], axis=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split data
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]
        meta_tr, meta_val = meta_full[train_idx], meta_full[val_idx]

        # Create Datasets
        # Enable augmentation (transform=True) only for training
        train_ds = IcebergDataset(X_tr, meta_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_val, meta_val, y_val, transform=False)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        loaders.append((train_loader, val_loader))

    return loaders


def get_test_loader(data_dict, batch_size=32):
    """
    Generates a DataLoader for the test set.

    Args:
        data_dict (dict): Dictionary containing 'X_test', 'meta_test'.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Iterator for test data.
    """
    test_ds = IcebergDataset(
        data_dict["X_test"], data_dict["meta_test"], y=None, transform=False
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return test_loader

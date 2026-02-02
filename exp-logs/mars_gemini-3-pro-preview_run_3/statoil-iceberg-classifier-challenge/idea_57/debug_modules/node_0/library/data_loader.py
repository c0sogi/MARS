import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import load_dataset, set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Handles 3-channel images (HH, HV, Avg) and scalar incidence angles.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        # Input is float32, so we just convert to tensor.
        # Note: torchvision transforms usually expect (C, H, W) tensors or PIL images.
        img_tensor = torch.from_numpy(img)

        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to tensor (keep as float32)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label = self.y[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


def get_loaders(fold_idx):
    """
    Generates train and validation loaders for a specific fold in 5-Fold CV.
    Combines the pre-split 'train' and 'val' metadata to perform a fresh K-Fold split.

    Args:
        fold_idx (int): The index of the fold (0 to N_FOLDS-1).

    Returns:
        tuple: (train_loader, val_loader)
    """
    set_seed(Config.SEED)

    # 1. Load all labeled data (Train + Val from metadata)
    # We ignore the fixed split in metadata and merge them for K-Fold
    X_train_part, ang_train_part, y_train_part, _ = load_dataset(
        "train", load_cached_data=True
    )
    X_val_part, ang_val_part, y_val_part, _ = load_dataset("val", load_cached_data=True)

    # Concatenate
    X = np.concatenate([X_train_part, X_val_part], axis=0)
    angles = np.concatenate([ang_train_part, ang_val_part], axis=0)
    y = np.concatenate([y_train_part, y_val_part], axis=0)

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(X, y))
    if fold_idx < 0 or fold_idx >= Config.N_FOLDS:
        raise ValueError(f"fold_idx must be between 0 and {Config.N_FOLDS - 1}")

    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train, X_val = X[train_idx], X[val_idx]
    ang_train, ang_val = angles[train_idx], angles[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 3. Define Transforms
    # RandomHorizontalFlip and RandomVerticalFlip work on tensors (C, H, W)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No transforms for validation
    val_transform = None

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=val_transform)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Generates the test loader for inference.

    Returns:
        DataLoader: The test data loader.
    """
    # 1. Load test data
    X_test, ang_test, _, ids_test = load_dataset("test", load_cached_data=True)

    # 2. Create Dataset (No transforms, no labels)
    test_dataset = IcebergDataset(X_test, ang_test, y=None, transform=None)

    # 3. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return test_loader

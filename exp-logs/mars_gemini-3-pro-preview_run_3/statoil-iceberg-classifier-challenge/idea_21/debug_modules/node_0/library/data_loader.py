import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.utils import load_data


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = torch.from_numpy(X).float()
        self.angles = torch.from_numpy(angles).float().unsqueeze(1)  # (N, 1)

        if y is not None:
            self.y = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1)
        else:
            self.y = None

        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = self.y[idx]
            return img, angle, label
        else:
            # For test set, return ID if available
            id_val = self.ids[idx] if self.ids is not None else ""
            return img, angle, id_val


def get_loaders(
    batch_size=32, n_splits=5, seed=42, debug=False, cache_dir="./working/idea_21"
):
    """
    Creates Stratified K-Fold DataLoaders for training and validation.

    Args:
        batch_size (int): Batch size.
        n_splits (int): Number of folds for Cross-Validation.
        seed (int): Random seed.
        debug (bool): If True, uses a small subset of data.
        cache_dir (str): Directory to store/load cached numpy arrays.

    Returns:
        list: A list of tuples (train_loader, val_loader) for each fold.
    """
    # Load data using the provided utility
    # This handles caching and preprocessing (reshaping, imputation, channel creation)
    (
        X_train_part,
        y_train_part,
        angles_train_part,
        ids_train_part,
        X_val_part,
        y_val_part,
        angles_val_part,
        ids_val_part,
        _,
        _,
        _,
    ) = load_data(cache_dir=cache_dir)

    # Combine the fixed train/val splits from metadata to perform full K-Fold CV
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    angles_full = np.concatenate([angles_train_part, angles_val_part], axis=0)

    if debug:
        # Use a small subset for debugging
        subset_size = min(100, len(X_full))
        indices = np.random.RandomState(seed).choice(
            len(X_full), subset_size, replace=False
        )
        X_full = X_full[indices]
        y_full = y_full[indices]
        angles_full = angles_full[indices]

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split data
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        angles_train_fold = angles_full[train_idx]

        X_val_fold = X_full[val_idx]
        y_val_fold = y_full[val_idx]
        angles_val_fold = angles_full[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold, angles_train_fold, y_train_fold, transform=train_transform
        )

        val_dataset = IcebergDataset(
            X_val_fold, angles_val_fold, y_val_fold, transform=None
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders


def get_test_loader(batch_size=32, cache_dir="./working/idea_21"):
    """
    Creates a DataLoader for the test set.

    Args:
        batch_size (int): Batch size.
        cache_dir (str): Directory to store/load cached numpy arrays.

    Returns:
        DataLoader: Test data loader returning (images, angles, ids).
    """
    # Load only test data
    # We unpack the tuple returned by load_data.
    # The last 3 elements are relevant for test.
    data_tuple = load_data(cache_dir=cache_dir)
    X_test = data_tuple[8]
    angles_test = data_tuple[9]
    ids_test = data_tuple[10]

    test_dataset = IcebergDataset(
        X_test, angles_test, y=None, ids=ids_test, transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader

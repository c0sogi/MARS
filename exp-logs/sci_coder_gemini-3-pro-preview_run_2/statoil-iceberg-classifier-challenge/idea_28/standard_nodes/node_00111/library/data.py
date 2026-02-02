import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import load_data


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Ship vs Iceberg classification.
    Handles 3-channel radar images and incidence angles.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32 (0 or 1).
            transform (bool): Whether to apply random augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]  # Scalar

        if self.transform:
            # Random Rotation: 0, 90, 180, 270 degrees
            # axes=(1, 2) corresponds to (H, W) for (C, H, W) format
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2))

            # Random Horizontal Flip
            # axis=2 corresponds to Width
            if np.random.random() < 0.5:
                image = np.flip(image, axis=2)

        # Ensure memory is contiguous after numpy operations (flip/rot)
        # to avoid stride issues when converting to Tensor
        image = np.ascontiguousarray(image)

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            return image_tensor, angle_tensor


def get_dataloaders(config, fold_index=0, debug=False):
    """
    Creates DataLoaders for a specific fold in Stratified K-Fold CV.

    Implements:
    - Strict fold-wise Min-Max scaling (stats derived from train fold only).
    - Incidence angle imputation (mean derived from train fold only).
    - Augmentation for training set.

    Args:
        config (Config): Configuration object.
        fold_index (int): Index of the fold (0 to N_FOLDS-1).
        debug (bool): If True, subsets data for rapid testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # 1. Load Data (uses caching mechanism in library.utils)
    data = load_data(config)

    all_train_images = data["train_images"]  # (N, 3, 75, 75)
    all_train_labels = data["train_labels"]  # (N,)
    all_train_angles = data["train_angles"]  # (N,)
    all_train_ids = data["train_ids"]  # (N,)

    test_images = data["test_images"]  # (M, 3, 75, 75)
    test_angles = data["test_angles"]  # (M,)
    test_ids = data["test_ids"]  # (M,)

    # Debug Mode: Subset data
    if debug:
        subset_size = 100
        all_train_images = all_train_images[:subset_size]
        all_train_labels = all_train_labels[:subset_size]
        all_train_angles = all_train_angles[:subset_size]

        test_subset = 50
        test_images = test_images[:test_subset]
        test_angles = test_angles[:test_subset]
        test_ids = test_ids[:test_subset]

    # 2. Stratified K-Fold Split
    # We split indices based on the full training set
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Generate splits and select the requested fold
    folds = list(skf.split(all_train_images, all_train_labels))
    train_idx, val_idx = folds[fold_index]

    # Extract Train/Val subsets
    X_train = all_train_images[train_idx]
    y_train = all_train_labels[train_idx]
    a_train = all_train_angles[train_idx]

    X_val = all_train_images[val_idx]
    y_val = all_train_labels[val_idx]
    a_val = all_train_angles[val_idx]

    # 3. Strict Fold-wise Preprocessing

    # A. Min-Max Scaling
    # Calculate stats ONLY on X_train to prevent leakage
    # Shape: (N, 3, 75, 75) -> min/max over (N, H, W) -> (3,)
    # Reshape to (1, 3, 1, 1) for broadcasting
    min_stat = np.min(X_train, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
    max_stat = np.max(X_train, axis=(0, 2, 3)).reshape(1, 3, 1, 1)

    range_stat = max_stat - min_stat
    # Avoid division by zero
    range_stat[range_stat == 0] = 1.0

    # Apply scaling (No hard clipping, outliers allowed)
    X_train = (X_train - min_stat) / range_stat
    X_val = (X_val - min_stat) / range_stat
    # Scale test set using training stats
    X_test = (test_images - min_stat) / range_stat

    # B. Incidence Angle Imputation
    # Calculate mean ONLY on X_train (ignoring NaNs)
    angle_mean = np.nanmean(a_train)

    # Fill NaNs
    a_train = np.nan_to_num(a_train, nan=angle_mean)
    a_val = np.nan_to_num(a_val, nan=angle_mean)
    a_test = np.nan_to_num(test_angles, nan=angle_mean)

    # 4. Create Datasets
    # Train: Transform=True (Augmentation)
    train_ds = IcebergDataset(X_train, a_train, y_train, transform=True)
    # Val: Transform=False
    val_ds = IcebergDataset(X_val, a_val, y_val, transform=False)
    # Test: Transform=False, No labels
    test_ds = IcebergDataset(X_test, a_test, labels=None, transform=False)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.model import GLiClassDataset, get_transforms


def get_loaders(fold=None):
    """
    Constructs and returns DataLoaders for training and validation.

    Args:
        fold (int, optional): The fold index (0-4) for Cross-Validation.
                              If None, uses the fixed Train/Val split defined in metadata.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # ------------------------------------------------------------------
    # Option 1: Fixed Split (Default)
    # Uses the pre-generated train_metadata.csv and val_metadata.csv
    # ------------------------------------------------------------------
    if fold is None:
        train_dataset = GLiClassDataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            split="train",
            transform=get_transforms("train"),
            load_cached_data=True,
        )

        val_dataset = GLiClassDataset(
            metadata_path=Config.VAL_METADATA_PATH,
            split="val",
            transform=get_transforms("val"),
            load_cached_data=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Option 2: GroupKFold Cross-Validation
    # Splits the training data dynamically based on Subject ID
    # ------------------------------------------------------------------
    else:
        # We need two instances of the dataset:
        # 1. With training augmentations
        # 2. With validation transforms (no augmentation)
        # Both read from the same cache/metadata.
        train_ds_aug = GLiClassDataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            split="train",
            transform=get_transforms("train"),
            load_cached_data=True,
        )

        val_ds_noaug = GLiClassDataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            split="train",
            transform=get_transforms("val"),
            load_cached_data=True,
        )

        # Extract groups (Subject IDs) for stratification
        # Note: dataset.ids contains the ID for each slice.
        # Since we have 3 slices per subject, multiple indices share the same ID.
        groups = train_ds_aug.ids
        y = train_ds_aug.targets
        X = np.zeros(len(y))  # Dummy features for splitter

        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(X, y, groups))

        train_idx, val_idx = folds[fold]

        # Create Subsets
        train_subset = Subset(train_ds_aug, train_idx)
        val_subset = Subset(val_ds_noaug, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader


def get_test_loader():
    """
    Constructs and returns the DataLoader for the test set.
    """
    test_dataset = GLiClassDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split="test",
        transform=get_transforms("test"),
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader

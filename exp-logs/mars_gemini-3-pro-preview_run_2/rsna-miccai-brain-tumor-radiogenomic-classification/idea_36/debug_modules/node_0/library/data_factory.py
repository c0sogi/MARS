import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    SEED,
)
from library.dicom_utils import load_and_preprocess_dataset


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )
    return None


class MemoryDataset(Dataset):
    """
    A PyTorch Dataset that wraps pre-loaded numpy arrays.
    Handles channel-first/channel-last conversion for Albumentations.
    """

    def __init__(self, data, targets, transforms=None, is_test=False):
        """
        Args:
            data (np.ndarray): Input data of shape (N, Channels, H, W).
            targets (np.ndarray): Targets (labels or IDs) of shape (N,).
            transforms (A.Compose): Albumentations transforms.
            is_test (bool): If True, returns ID as long; else returns label as float.
        """
        self.data = data
        self.targets = targets
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve data: (Channels, H, W)
        img = self.data[idx]
        target = self.targets[idx]

        # Convert to (H, W, Channels) for Albumentations
        img_np = np.transpose(img, (1, 2, 0))

        # Apply transforms if available
        if self.transforms:
            augmented = self.transforms(image=img_np)
            img_np = augmented["image"]

        # Convert back to (Channels, H, W) and create tensor
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1)))

        if self.is_test:
            # For test set, target is the BraTS21ID (int)
            return img_tensor, torch.tensor(target, dtype=torch.long)
        else:
            # For train/val, target is MGMT_value (float for BCE)
            return img_tensor, torch.tensor(target, dtype=torch.float32)


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates the loading of Train, Validation, and Test datasets.
    Uses library.dicom_utils.load_and_preprocess_dataset for processing and caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from disk cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # 2. Process/Load Data using the utility with Circuit Breaker & Caching
    # The utility handles the 'working/idea_36' directory and .npy files
    print("Initializing Data Pipeline...")

    train_data, train_labels = load_and_preprocess_dataset(
        train_df, "train", INPUT_DIR, load_cached_data=load_cached_data
    )

    val_data, val_labels = load_and_preprocess_dataset(
        val_df, "val", INPUT_DIR, load_cached_data=load_cached_data
    )

    test_data, test_ids = load_and_preprocess_dataset(
        test_df, "test", INPUT_DIR, load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    train_ds = MemoryDataset(
        train_data, train_labels, transforms=get_transforms("train"), is_test=False
    )

    val_ds = MemoryDataset(
        val_data, val_labels, transforms=get_transforms("val"), is_test=False
    )

    test_ds = MemoryDataset(
        test_data, test_ids, transforms=get_transforms("test"), is_test=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

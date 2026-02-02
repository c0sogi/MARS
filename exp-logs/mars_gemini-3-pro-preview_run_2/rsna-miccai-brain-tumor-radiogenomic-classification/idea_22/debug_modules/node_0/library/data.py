import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library import config, utils


class MGMTDataset(Dataset):
    def __init__(self, X, y, augment=False):
        """
        Args:
            X (np.ndarray): Input images of shape (N, 12, 224, 224).
            y (np.ndarray): Labels of shape (N,).
            augment (bool): Whether to apply training augmentations.
        """
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape: (12, 224, 224)
        img = self.X[idx].copy()
        label = self.y[idx]

        if self.augment:
            # Transpose to (H, W, C) for OpenCV operations
            # (12, 224, 224) -> (224, 224, 12)
            img_HWC = np.transpose(img, (1, 2, 0))

            # 1. Horizontal Flip
            if np.random.rand() < 0.5:
                img_HWC = np.fliplr(img_HWC)

            # 2. Vertical Flip
            if np.random.rand() < 0.5:
                img_HWC = np.flipud(img_HWC)

            # 3. Rotation (+/- 15 degrees)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img_HWC.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Ensure contiguous array for cv2.warpAffine after flips
                img_HWC = np.ascontiguousarray(img_HWC)

                # warpAffine handles multi-channel images correctly
                # borderMode=cv2.BORDER_CONSTANT, borderValue=0 gives zero padding
                img_HWC = cv2.warpAffine(
                    img_HWC,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

            # Transpose back to (C, H, W)
            # (224, 224, 12) -> (12, 224, 224)
            img = np.transpose(img_HWC, (2, 0, 1))

        # Convert to tensor
        # Input is already float32 from utils
        img_tensor = torch.from_numpy(np.ascontiguousarray(img))

        # Label handling
        if label != -1:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(-1.0, dtype=torch.float32)

        return img_tensor, label_tensor


def get_data_loaders(load_cached_data=True):
    """
    Loads data using library.utils and returns DataLoaders for train, val, and test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA)
    df_val = pd.read_csv(config.VAL_METADATA)
    df_test = pd.read_csv(config.TEST_METADATA)

    # 2. Load/Process Data (using utils caching mechanism)
    # The utils.get_dataset function handles the caching logic (check/load/save)
    # It returns numpy arrays X and y
    X_train, y_train = utils.get_dataset(
        df_train, "train", load_cached_data=load_cached_data
    )
    X_val, y_val = utils.get_dataset(df_val, "val", load_cached_data=load_cached_data)
    X_test, y_test = utils.get_dataset(
        df_test, "test", load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    # Apply augmentations only to training set
    train_dataset = MGMTDataset(X_train, y_train, augment=True)
    val_dataset = MGMTDataset(X_val, y_val, augment=False)
    test_dataset = MGMTDataset(X_test, y_test, augment=False)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to ensure stable Batch Norm statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

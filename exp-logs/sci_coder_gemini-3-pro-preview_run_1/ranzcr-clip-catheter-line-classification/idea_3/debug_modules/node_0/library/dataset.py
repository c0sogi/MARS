import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter Detection.
    Handles loading images, converting to RGB, and applying augmentations.
    """

    def __init__(self, df, transforms=None, input_dir=Config.input_dir):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_paths, labels).
            transforms (albumentations.Compose): Albumentations transformations.
            input_dir (str): Root directory for input images.
        """
        self.df = df
        self.transforms = transforms
        self.input_dir = input_dir

        # Pre-extract paths and labels for faster access
        self.file_paths = df["file_path"].values

        # Extract labels if target columns exist in the dataframe
        # For test set, these might be placeholders, which is fine
        self.labels = df[Config.target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # cv2.imread returns BGR or Grayscale
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # Robustness check
        if image is None:
            # Return a black image if file is corrupt/missing (should be caught by metadata check)
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)

        # Ensure 3 channels (RGB)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def get_transforms(data="train"):
    """
    Returns the albumentations transformation pipeline.

    Args:
        data (str): 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # CoarseDropout to encourage distributed feature learning for thin lines
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(f"Unknown data mode: {data}")


def get_dataloaders(train_df, val_df, test_df, batch_size=Config.batch_size):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create Datasets
    train_ds = CatheterDataset(train_df, transforms=get_transforms("train"))

    val_ds = CatheterDataset(val_df, transforms=get_transforms("valid"))

    test_ds = CatheterDataset(test_df, transforms=get_transforms("valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

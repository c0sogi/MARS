import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import get_weighted_sampler


class HerbariumDataset(Dataset):
    """
    Custom Dataset for Herbarium Plant Species Classification.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, category_id/image_id).
            transforms (albumentations.Compose): Albumentations transforms to apply.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-fetch paths to avoid overhead in __getitem__
        self.file_paths = df["file_path"].values

        # Pre-fetch labels or image_ids depending on mode
        if self.mode != "test":
            self.labels = df["category_id"].values
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct absolute file path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        # Default imread loads as BGR with 3 channels
        image = cv2.imread(full_path)

        # Robustness check: if image fails to load, return a blank image
        if image is None:
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode != "test":
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            image_id = self.image_ids[idx]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_transforms(img_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        img_size (int): Target image size (height and width).
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(train_df, val_df, test_df, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        load_cached_data (bool): Whether to use cached weights for the sampler.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Handle Debug Mode: Slice datasets if enabled
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Truncating datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        # Test DF might be needed fully for submission structure, but for debug loop usually we slice it too
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    # Define Transforms
    train_transforms = get_transforms(Config.IMG_SIZE, mode="train")
    val_transforms = get_transforms(Config.IMG_SIZE, mode="val")

    # Initialize Datasets
    train_dataset = HerbariumDataset(
        train_df, transforms=train_transforms, mode="train"
    )
    val_dataset = HerbariumDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = HerbariumDataset(test_df, transforms=val_transforms, mode="test")

    # Initialize Weighted Sampler for Training
    # This ensures rare classes are seen frequently
    train_sampler = get_weighted_sampler(train_df, load_cached_data=load_cached_data)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

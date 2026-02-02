import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(
        self, df: pd.DataFrame, root_dir: str, transform=None, labeled: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            root_dir (str): Root directory where images are stored.
            transform (albumentations.Compose, optional): Augmentation pipeline.
            labeled (bool): Whether the dataset contains labels (True for train/val, False for test).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled

        # Pre-extract file paths and labels to avoid overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        if self.labeled:
            self.labels = self.df[Config.LABEL_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # The metadata contains relative paths like "images/Train_0.jpg"
        # Config.INPUT_DIR is "./input"
        # So we join input_dir with the relative path
        file_path = os.path.join(self.root_dir, self.file_paths[idx])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.labeled:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(data: str, image_size: int):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train' or 'valid'/'test'.
        image_size (int): Target spatial dimension (e.g., 380 or 224).
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                # Strong Geometric Augmentations
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Wide scaling limits as per idea description
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=0.5
                ),
                # Normalization and Tensor Conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


def get_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, image_size: int, batch_size: int
):
    """
    Creates and returns DataLoaders for training and validation.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        image_size (int): Image resolution for resizing.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    train_transforms = get_transforms("train", image_size)
    val_transforms = get_transforms("valid", image_size)

    # Instantiate Datasets
    # Note: Config.INPUT_DIR is used as root because file_paths in metadata are relative (e.g., "images/Test_0.jpg")
    train_dataset = AppleDataset(
        df=train_df, root_dir=Config.INPUT_DIR, transform=train_transforms, labeled=True
    )

    val_dataset = AppleDataset(
        df=val_df, root_dir=Config.INPUT_DIR, transform=val_transforms, labeled=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(test_df: pd.DataFrame, image_size: int, batch_size: int):
    """
    Creates and returns a DataLoader for the test set.

    Args:
        test_df (pd.DataFrame): Test metadata.
        image_size (int): Image resolution.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test data loader.
    """
    test_transforms = get_transforms("test", image_size)

    test_dataset = AppleDataset(
        df=test_df, root_dir=Config.INPUT_DIR, transform=test_transforms, labeled=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

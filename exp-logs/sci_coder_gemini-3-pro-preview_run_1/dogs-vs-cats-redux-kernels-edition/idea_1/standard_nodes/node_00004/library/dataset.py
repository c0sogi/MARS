import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.utils import Config


class DogCatDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(self, df, input_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            input_dir (str): Root directory for input images.
            transform (A.Compose): Albumentations transforms pipeline.
            is_test (bool): Flag to indicate if this is the test set (returns id instead of label).
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Read image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission
            return image, row["id"]
        else:
            # Return image and label (float32 for BCEWithLogitsLoss)
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label


def get_transforms(phase, img_size=224):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        # Cite solution_lesson_node_00003: Stronger augmentation for larger model capacity
        return A.Compose(
            [
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test phases
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def create_dataloaders(config: Config):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load Metadata
    df_train = pd.read_csv(config.train_metadata)
    df_val = pd.read_csv(config.val_metadata)
    df_test = pd.read_csv(config.test_metadata)

    # Debug Mode: Sample data
    if config.debug:
        df_train = df_train.head(config.debug_sample_size)
        df_val = df_val.head(config.debug_sample_size)
        df_test = df_test.head(config.debug_sample_size)
        print(f"Debug mode enabled. Reduced train size to {len(df_train)}")

    # Define Transforms
    train_transform = get_transforms("train", config.img_size)
    val_transform = get_transforms("val", config.img_size)
    test_transform = get_transforms("test", config.img_size)

    # Create Datasets
    train_dataset = DogCatDataset(
        df_train, config.input_dir, transform=train_transform, is_test=False
    )
    val_dataset = DogCatDataset(
        df_val, config.input_dir, transform=val_transform, is_test=False
    )
    test_dataset = DogCatDataset(
        df_test, config.input_dir, transform=test_transform, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

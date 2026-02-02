import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from timm.data.mixup import Mixup

from library.config import Config


class AnimalDataset(Dataset):
    """
    Custom Dataset for Animal Classification.
    Reads image paths and labels from metadata DataFrames.
    """

    def __init__(self, metadata_path, transform=None, is_test=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Transformations to apply to the image.
            is_test (bool): Whether this is the test set (no labels expected).
        """
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative path e.g., 'train_images/xyz.jpg'
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though EDA showed 0 missing)
            # Create a black image to avoid crashing
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Label
        if self.is_test:
            # For test set, return dummy label or just the image
            # Returning a dummy 0 to keep signature consistent
            target = torch.tensor(0, dtype=torch.long)
        else:
            target = torch.tensor(row["Category"], dtype=torch.long)

        return image, target


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    height, width = Config.IMG_SIZE

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                # Standard Augmentations to replace Mixup regularization
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_mixup_fn():
    """
    Creates the Mixup/CutMix callable using timm.

    Returns:
        Mixup: The mixup function to be applied to batches.
    """
    if not Config.USE_MIXUP_CUTMIX:
        return None

    return Mixup(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=Config.MIXUP_SWITCH_PROB,
        mode=Config.MIXUP_MODE,
        label_smoothing=0.0,  # Loss function handles smoothing if needed, or set here
        num_classes=Config.NUM_CLASSES,
    )


def get_dataloaders():
    """
    Creates DataLoaders for train, validation, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Datasets
    train_dataset = AnimalDataset(
        metadata_path=Config.TRAIN_META_PATH, transform=train_transform, is_test=False
    )

    val_dataset = AnimalDataset(
        metadata_path=Config.VAL_META_PATH, transform=val_transform, is_test=False
    )

    test_dataset = AnimalDataset(
        metadata_path=Config.TEST_META_PATH, transform=test_transform, is_test=True
    )

    # DataLoaders
    # Drop last for train to ensure consistent batch sizes for Mixup
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

import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_ROOT,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def get_transforms(phase):
    """
    Returns the data augmentation pipeline based on the phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Force model to recognize animals from partial views and varying scales
                A.RandomResizedCrop(
                    height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.5, 1.0), p=1.0
                ),
                # Simulate domain shift (lighting, contrast)
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # Simulate domain shift (blur/sharpness)
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                # Horizontal flip is generally safe for animals
                A.HorizontalFlip(p=0.5),
                # CoarseDropout (Cutout) to prevent reliance on single features/backgrounds
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(IMAGE_SIZE * 0.1),
                    max_width=int(IMAGE_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class CameraTrapDataset(Dataset):
    """
    Custom Dataset for Camera Trap Images.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Id, file_path, Category).
            root_dir (str): Root directory where images are stored.
            transform (callable, optional): Albumentations transform pipeline.
            is_test (bool): If True, returns dummy label or ignores label column.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input root (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing images (though metadata check passed)
            # Create a black image to avoid crashing
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (should not happen in this pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Get label
        if self.is_test:
            # For test set, return image and Id (for submission mapping)
            # We return a dummy label 0 for consistency in loops if needed,
            # but usually test loops just need the image.
            # However, returning the ID is crucial for submission.
            return image, row["Id"]
        else:
            label = row["Category"]
            return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load Metadata
    df_train = pd.read_csv(TRAIN_META_PATH)
    df_val = pd.read_csv(VAL_META_PATH)
    df_test = pd.read_csv(TEST_META_PATH)

    # Define Transforms
    train_transform = get_transforms(phase="train")
    val_transform = get_transforms(phase="val")
    test_transform = get_transforms(phase="test")

    # Create Datasets
    train_dataset = CameraTrapDataset(
        df_train, INPUT_ROOT, transform=train_transform, is_test=False
    )

    val_dataset = CameraTrapDataset(
        df_val, INPUT_ROOT, transform=val_transform, is_test=False
    )

    test_dataset = CameraTrapDataset(
        df_test, INPUT_ROOT, transform=test_transform, is_test=True
    )

    # Create DataLoaders
    # Note: We are NOT using WeightedRandomSampler as per the new strategy.
    # We rely on ClassBalancedFocalLoss and Strong Augmentation.

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

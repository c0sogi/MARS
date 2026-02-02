import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_train_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the training transformations including RandomResizedCrop and HorizontalFlip.

    Args:
        img_size (int): The target height and width for the image.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the validation/test transformations (Resize and Normalize).

    Args:
        img_size (int): The target height and width for the image.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class DogCatDataset(Dataset):
    """
    Custom Dataset for Dog vs Cat classification.
    Reads images using OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and 'label' (or 'id').
            transforms (albumentations.Compose): Transformations to apply to the image.
            mode (str): One of 'train', 'val', 'test'.
                        'train'/'val' returns (image, label).
                        'test' returns (image, id).
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepaths are relative to the input directory (e.g., "train/cat.0.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Check for missing or corrupt files
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupt at path: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor and normalize to [0, 1] if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return data based on mode
        if self.mode in ["train", "val"]:
            # Label is 0 (Cat) or 1 (Dog)
            # We return float32 for BCEWithLogitsLoss compatibility
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # Test mode requires ID for submission
            img_id = row["id"]
            return image, img_id

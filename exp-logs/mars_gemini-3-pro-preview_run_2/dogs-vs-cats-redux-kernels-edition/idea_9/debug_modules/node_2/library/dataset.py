import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_type (str): 'train' for training augmentations, 'valid' or 'test' for validation/inference.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=(Config.CROP_SCALE_MIN, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class PetDataset(Dataset):
    """
    Torch Dataset for Dog vs Cat classification.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train' (returns img, label) or 'test' (returns img, id).
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-check columns to ensure metadata validity
        if self.mode == "train":
            if "label" not in self.df.columns:
                raise ValueError("DataFrame must contain 'label' column for train mode")
        elif self.mode == "test":
            if "id" not in self.df.columns:
                raise ValueError("DataFrame must contain 'id' column for test mode")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # Metadata filepaths are relative to INPUT_DIR (e.g., "train/cat.0.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        if self.mode == "train":
            # Return image and label
            label = row["label"]
            # Convert label to float for BCEWithLogitsLoss / Mixup compatibility
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # Return image and id for submission mapping
            img_id = row["id"]
            return image, torch.tensor(img_id, dtype=torch.long)

import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the augmentation pipeline based on the phase (train/valid/test).

    Args:
        phase (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The albumentations composition of transforms.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=(0.08, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test: Resize short side to 256, then CenterCrop to 224
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=Config.RESIZE_SIZE),
                A.CenterCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class INatDataset(Dataset):
    """
    PyTorch Dataset for the iNaturalist 2019 competition.
    """

    def __init__(self, df: pd.DataFrame, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_name, category_id).
            transforms (A.Compose, optional): Albumentations transforms pipeline.
        """
        self.df = df
        self.transforms = transforms

        # Pre-extract columns to lists for faster access
        self.file_names = self.df["file_name"].values
        self.image_ids = self.df["image_id"].values

        # Handle targets
        if "category_id" in self.df.columns:
            self.targets = self.df["category_id"].values
        else:
            # For test set, fill with -1
            self.targets = np.full(len(self.df), -1, dtype=int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_id = self.image_ids[idx]
        target = self.targets[idx]

        # Construct full path
        file_path = os.path.join(Config.INPUT_DIR, file_name)

        # Load image using OpenCV
        image = cv2.imread(file_path)

        # Robustness check: if image fails to load, create a black image
        if image is None:
            # Create a blank image with the resize dimensions to avoid transform errors
            # Defaulting to RESIZE_SIZE to ensure SmallestMaxSize works as expected
            image = np.zeros(
                (Config.RESIZE_SIZE, Config.RESIZE_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transforms provided (should generally not happen in this pipeline)
            # Just convert to tensor
            image = ToTensorV2()(image=image)["image"]

        return (
            image,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(image_id, dtype=torch.long),
        )

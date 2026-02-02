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
    Returns the albumentations transformations for the specified data type.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transformations.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=Config.AUG_PROB),
                A.VerticalFlip(p=Config.AUG_PROB),
                A.ShiftScaleRotate(rotate_limit=Config.AUG_ROTATION, p=Config.AUG_PROB),
                A.RandomBrightnessContrast(p=Config.AUG_PROB),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Extract file paths (relative to input dir)
        self.file_paths = df["file_path"].values

        # Pre-extract targets or IDs to speed up __getitem__
        if self.mode != "test":
            # Ensure targets are extracted in the order defined in Config.CLASSES
            self.targets = df[Config.CLASSES].values
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # Metadata file_paths are relative, e.g., "images/Train_0.jpg"
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode != "test":
            # Return image and soft targets (probabilities) as float tensors
            target = self.targets[idx]
            return image, torch.tensor(target, dtype=torch.float32)
        else:
            # Return image and image_id for submission generation
            return image, self.image_ids[idx]

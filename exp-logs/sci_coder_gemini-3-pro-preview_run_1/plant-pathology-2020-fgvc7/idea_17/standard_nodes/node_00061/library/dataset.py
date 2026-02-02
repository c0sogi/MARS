import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading, color conversion, and label extraction.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, output_label: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, targets).
            transforms (albumentations.Compose): Albumentations transformations.
            output_label (bool): Whether to return labels (True for train/val, False for test).
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.output_label = output_label
        self.classes = Config.CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input dir (e.g., "images/Train_0.jpg")
        # Config.INPUT_DIR is "./input"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.output_label:
            # Fetch target labels for training/validation
            labels = row[self.classes].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # Return image and image_id for inference/submission
            return image, row["image_id"]


def get_transforms(data: str = "valid"):
    """
    Returns the albumentations transformation pipeline based on the data split.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Strategic augmentations based on Lesson 30 (VerticalFlip) and Lesson 3 (ShiftScaleRotate)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
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
    else:
        raise ValueError(f"Unknown data mode: {data}")

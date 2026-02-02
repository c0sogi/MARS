import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline based on the data mode.

    Args:
        data (str): Mode of operation ('train', 'valid', or 'test').

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                # Aggressive cropping to capture local disease features
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=(Config.AUG_SCALE_MIN, Config.AUG_SCALE_MAX),
                    p=1.0,
                ),
                # Rotational invariance
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                # Lighting invariance
                A.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=Config.AUG_COLOR_JITTER,
                    hue=Config.AUG_COLOR_JITTER,
                    p=0.5,
                ),
                # Standard Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms: Deterministic resizing
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading, augmentation, and multi-hot label encoding.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'image', 'labels', and 'file_path'.
            transforms (A.Compose, optional): Albumentations transforms.
        """
        self.df = df
        self.transforms = transforms
        self.classes = Config.CLASSES
        self.num_classes = Config.NUM_CLASSES
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct the full path to the image
        # row['file_path'] is relative, e.g., 'train_images/img.jpg'
        # Config.INPUT_DIR is ./input
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Safety check for missing images (though metadata check passed)
        if image is None:
            # Return a blank image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            image = transform(image=image)["image"]

        # Process Labels (Multi-hot encoding)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        labels_str = row.get("labels", "")

        # Parse space-delimited labels
        if isinstance(labels_str, str) and labels_str:
            for label in labels_str.split():
                if label in self.classes:
                    class_idx = self.classes.index(label)
                    target[class_idx] = 1.0

        return image, target

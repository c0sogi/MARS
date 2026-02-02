import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_split, img_size):
    """
    Returns the Albumentations transformation pipeline for a specific data split and image resolution.

    Args:
        data_split (str): One of 'train', 'val', or 'test'.
        img_size (int): The target height and width for resizing (e.g., 256 or 512).

    Returns:
        A.Compose: The composed albumentations transform.
    """
    # Common normalization statistics (ImageNet defaults)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                # Augmentations as specified in the strategy
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data_split in ["val", "test"]:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data_split: {data_split}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Reads images from disk and applies transformations.
    """

    def __init__(self, metadata, transform=None, output_extra=False):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing image paths and labels (for train/val).
            transform (A.Compose, optional): Albumentations transforms to apply.
            output_extra (bool): If True, returns image_id in the output dict (useful for inference).
        """
        self.metadata = metadata
        self.transform = transform
        self.output_extra = output_extra

        # Check if target columns exist in metadata
        self.target_cols = Config.TARGET_COLS
        self.has_labels = all(col in self.metadata.columns for col in self.target_cols)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get row
        row = self.metadata.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to input dir (e.g., "images/Train_0.jpg")
        # Config.INPUT_DIR is "./input"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare output
        result = {"image": image}

        # Add labels if available
        if self.has_labels:
            # Get labels as float32 for BCEWithLogitsLoss
            labels = row[self.target_cols].values.astype(np.float32)
            result["target"] = torch.tensor(labels, dtype=torch.float32)

        # Add extra info if requested (e.g. for submission file creation)
        if self.output_extra:
            result["image_id"] = row["image_id"]

        return result

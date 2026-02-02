import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transformation pipeline based on the data split.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                # Resize to ensure input consistency
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Augmentations specified in Idea 5
                # Explicitly ensuring HorizontalFlip is included
                A.HorizontalFlip(p=0.5),
                # Wide rotation range and shifting/scaling
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=45,  # Wide rotation range
                    p=0.5,
                ),
                # Brightness and Contrast adjustments
                A.RandomBrightnessContrast(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    elif data in ["valid", "test"]:
        return A.Compose(
            [
                # Deterministic resizing for validation/testing
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Normalization matching training stats
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Unknown data mode: {data}. Expected 'train', 'valid', or 'test'."
        )


class AppleLeafDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Reads images based on metadata paths and applies transformations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, labels).
            transforms (A.Compose): Albumentations transforms to apply.
            mode (str): 'train', 'valid', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract file paths and labels (if available) for faster access
        self.file_paths = self.df["file_path"].values

        if self.mode != "test":
            # Extract target columns for training/validation
            self.labels = self.df[Config.CLASS_LABELS].values.astype(np.float32)
        else:
            self.image_ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # Metadata paths are relative to input dir (e.g., "images/Train_0.jpg")
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should not happen based on metadata checks)
            # Create a black image of expected size to prevent crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            return image, self.image_ids[idx]
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

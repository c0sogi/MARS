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
    Returns the Albumentations transformations for the specified data split.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data == "train":
        # Heavy Augmentation strategy:
        # 1. RandomResizedCrop: Forces model to look at different parts/scales
        # 2. ShiftScaleRotate: Adds geometric invariance
        # 3. ColorJitter: Adds photometric invariance
        # 4. HorizontalFlip: Standard for natural images
        # Note: Mixup and CutMix are excluded as per requirements.
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.image_size, Config.image_size),
                    scale=(0.8, 1.0),
                    p=1.0,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    elif data == "valid" or data == "test":
        # Deterministic processing for evaluation
        return A.Compose(
            [
                A.Resize(height=Config.image_size, width=Config.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepath is relative to input_dir (e.g., "train/cat.0.jpg")
        img_path = os.path.join(Config.input_dir, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode in ["train", "val"]:
            # Return image and label
            label = row["label"]
            # Ensure label is float for BCEWithLogitsLoss (binary classification)
            return image, torch.tensor(label, dtype=torch.float32)

        elif self.mode == "test":
            # Return image and id for submission mapping
            file_id = row["id"]
            return image, file_id
        else:
            # Fallback
            return image

import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                # Geometric Intensity: Explicitly including VerticalFlip and ShiftScaleRotate
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=45,  # Wide rotation limit as requested
                    p=0.5,
                ),
                # Normalization for ImageNet pre-trained models
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Loads images and returns them with soft-target probability vectors.
    """

    def __init__(self, df: pd.DataFrame, transform=None, test_mode: bool = False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (image_id, file_path, targets).
            transform (albumentations.Compose, optional): Transforms to apply to the images.
            test_mode (bool): If True, returns only the image tensor (no targets).
        """
        self.df = df
        self.transform = transform
        self.test_mode = test_mode

        # Pre-compute full file paths
        # Metadata 'file_path' is relative (e.g., "images/Train_0.jpg")
        # Config.input_dir is "./input"
        self.file_paths = [
            os.path.join(Config.input_dir, fp) for fp in df["file_path"].values
        ]

        # Extract targets if not in test mode
        if not self.test_mode:
            # Ensure we are using the correct target columns defined in Config
            # We convert to float32 to support Soft-Target Cross-Entropy
            self.targets = df[Config.target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.test_mode:
            return image
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, target

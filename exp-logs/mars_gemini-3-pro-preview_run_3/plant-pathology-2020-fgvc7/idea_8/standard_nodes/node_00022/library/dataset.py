import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(data: str, img_size: int):
    """
    Constructs the Albumentations transformation pipeline based on the data split and model requirements.

    Args:
        data (str): The data split ('train', 'val', 'test').
        img_size (int): The target resolution for the image (e.g., 380 or 224).

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentation per strategy
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,  # Wide scaling limits
                    rotate_limit=45,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                # Photometric augmentations and Cutout are strictly excluded per strategy
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "val" or data == "test":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for loading Apple Leaf images and corresponding disease labels.
    """

    def __init__(self, df: pd.DataFrame, transform=None, return_label: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image IDs and metadata.
            transform (albumentations.Compose, optional): Augmentation pipeline.
            return_label (bool): If True, returns (image, label). If False, returns (image).
        """
        self.df = df
        self.transform = transform
        self.return_label = return_label
        self.file_paths = df["file_path"].values

        # Extract labels if required
        if self.return_label:
            # Verify all class columns exist in the dataframe
            missing_cols = [col for col in Config.CLASSES if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"The following class columns are missing from metadata: {missing_cols}"
                )

            # Store labels as a numpy array of shape (N, Num_Classes)
            self.labels = df[Config.CLASSES].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve full file path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data
        if self.return_label:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_dataset(split: str, img_size: int, debug: bool = False):
    """
    Factory function to initialize the AppleDataset with the correct metadata and transforms.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        img_size (int): Target image size for resizing.
        debug (bool): If True, limits the dataset to a small subset for debugging.

    Returns:
        AppleDataset: The initialized dataset.
    """
    # Determine configuration based on split
    if split == "train":
        csv_path = Config.TRAIN_CSV
        return_label = True
        transform_mode = "train"
    elif split == "val":
        csv_path = Config.VAL_CSV
        return_label = True
        transform_mode = "val"
    elif split == "test":
        csv_path = Config.TEST_CSV
        return_label = False
        transform_mode = "test"
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Load metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Apply debug subsetting
    if debug:
        df = df.head(50).copy()

    # Initialize transforms
    transforms = get_transforms(transform_mode, img_size)

    # Create dataset
    dataset = AppleDataset(df, transform=transforms, return_label=return_label)

    return dataset

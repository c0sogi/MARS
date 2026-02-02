import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter Detection on Chest X-rays.
    Handles loading images, converting to RGB, and applying augmentations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: A.Compose = None,
        mode: str = "train",
        debug: bool = False,
        debug_size: int = 100,
    ):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing metadata (StudyInstanceUID, file_path, labels).
            transforms (A.Compose, optional): Albumentations transform pipeline.
            mode (str): 'train', 'valid', or 'test'.
            debug (bool): If True, limits the dataset to `debug_size` samples.
            debug_size (int): Number of samples to use in debug mode.
        """
        # Apply debug slicing if requested
        if debug:
            self.df = df.iloc[:debug_size].reset_index(drop=True)
        else:
            self.df = df

        self.transforms = transforms
        self.mode = mode

        # Pre-fetch file paths to avoid overhead during __getitem__
        # file_path in metadata is relative to Config.INPUT_DIR
        self.file_paths = self.df["file_path"].values

        if self.mode != "test":
            # Extract labels for training/validation
            # Ensure columns follow the order specified in Config
            self.labels = self.df[Config.TARGET_COLS].values
        else:
            self.labels = None

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Fetches a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) if mode is not 'test', otherwise (image).
        """
        # Construct full path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        # cv2.imread loads in BGR format
        image = cv2.imread(img_path)

        # Robustness check: if image fails to load, return a blank image
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(data: str = "train") -> A.Compose:
    """
    Generates the Albumentations transform pipeline based on the data split.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composed augmentation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # Resize to high resolution (768x768) as per idea
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                # Horizontal Flip for TTA consistency and data augmentation
                A.HorizontalFlip(p=0.5 if Config.HORIZONTAL_FLIP else 0.0),
                # CoarseDropout for regularization
                # Helps the model focus on distributed features rather than single cues
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMAGE_SIZE * 0.1),
                    max_width=int(Config.IMAGE_SIZE * 0.1),
                    min_holes=4,
                    min_height=int(Config.IMAGE_SIZE * 0.05),
                    min_width=int(Config.IMAGE_SIZE * 0.05),
                    fill_value=0,
                    p=1.0 if Config.USE_COARSE_DROPOUT else 0.0,
                ),
                # Normalize using ImageNet statistics
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
                # Resize
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(
            f"Unknown data mode: {data}. Expected 'train', 'valid', or 'test'."
        )

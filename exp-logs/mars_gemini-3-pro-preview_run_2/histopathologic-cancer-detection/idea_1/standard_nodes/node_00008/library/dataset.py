import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from library.config import Config


def get_transforms(split: str):
    """
    Constructs the data transformation pipeline based on the dataset split.

    Args:
        split (str): The dataset split ('train', 'val', or 'test').

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    # Base transformations applied to all splits
    # 1. Convert numpy array (from cv2) to PIL Image
    transforms_list = [
        T.ToPILImage(),
    ]

    # Augmentations for training only
    # Applied BEFORE cropping to avoid artifacts and improve context - Cite Lesson 00004
    if split == "train":
        transforms_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=180),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
        )

    # 2. Center crop to the configured size (64x64)
    transforms_list.append(T.CenterCrop(Config.CROP_SIZE))

    # Final conversions for all splits
    # 1. Convert to Tensor (scales to [0, 1])
    # 2. Normalize using ImageNet mean and std
    transforms_list.extend(
        [T.ToTensor(), T.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)]
    )

    return T.Compose(transforms_list)


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for loading pathology images and labels.
    """

    def __init__(self, metadata_path, transform=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (callable, optional): Transform to be applied on a sample.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.metadata_path = metadata_path
        self.transform = transform

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Handle debug mode
        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths (e.g., 'train/{id}.tif')
        # Config.INPUT_DIR points to './input'
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Check if image loaded successfully
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Get label
        # BCEWithLogitsLoss requires float targets
        label = torch.tensor(row["label"], dtype=torch.float32)

        return img, label

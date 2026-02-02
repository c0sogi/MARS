import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data_split: str):
    """
    Constructs the transformation pipeline for a specific data split.

    Args:
        data_split (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composed albumentations transformations.
    """
    # Base transformations applied to all splits
    # Resize to the target size defined in Config
    # Normalize using ImageNet mean and std
    # Convert to PyTorch Tensor
    transforms_list = [
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
        A.Normalize(mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ]

    # Augmentations for training set only
    if data_split == "train":
        # Insert RandomHorizontalFlip before Normalization
        # Index 1 places it after Resize and before Normalize
        transforms_list.insert(1, A.HorizontalFlip(p=0.5))

    return A.Compose(transforms_list)


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for loading Cassava leaf images and corresponding labels.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, output_label: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'image_id', 'label', and 'file_path'.
            transforms (albumentations.Compose, optional): Transformations to apply to the images.
            output_label (bool): If True, returns (image, label). If False, returns (image).
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata for the current index
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to the input root (e.g., "train_images/xyz.jpg")
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(full_path)

        # Basic validation
        if image is None:
            raise FileNotFoundError(
                f"Image not found or could not be read at: {full_path}"
            )

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic
        if self.output_label:
            # Return image and label (as long tensor)
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            # Return only image (useful for inference if labels are missing/ignored)
            return image


def get_dataset(split: str, debug: bool = Config.DEBUG):
    """
    Factory function to load metadata and create a CassavaDataset instance.

    Args:
        split (str): The dataset split to load ('train', 'val', 'test').
        debug (bool): If True, loads only a small subset of data for debugging.

    Returns:
        CassavaDataset: The configured dataset instance.
    """
    # Determine metadata path and configuration based on split
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        output_label = True
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        output_label = True
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
        # For test, we typically don't need labels for the model input,
        # but the dataset class can handle it if they exist (placeholder '4').
        # We set output_label=False to return just the image for inference loops.
        output_label = False
    else:
        raise ValueError(
            f"Invalid split provided: {split}. Expected 'train', 'val', or 'test'."
        )

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Apply debug slicing if requested
    if debug:
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

    # Get appropriate transforms
    transforms = get_transforms(split)

    # Instantiate and return the dataset
    dataset = CassavaDataset(df=df, transforms=transforms, output_label=output_label)

    return dataset

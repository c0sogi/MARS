import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transformation pipeline for the specified data split.

    Args:
        data (str): 'train' for training augmentations, 'valid' or 'test' for validation/inference.

    Returns:
        A.Compose: The composition of transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Transpose(p=0.5),  # Cite solution_lesson_node_00027
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Geometric transformations as per strategy (Lesson 00004)
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
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
        raise ValueError(f"Unknown data type: {data}")


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, output_extra: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transforms (A.Compose, optional): Albumentations transforms.
            output_extra (bool): If True, returns (image, label, image_id).
                                 Useful for inference/debugging.
        """
        self.df = df
        self.transforms = transforms
        self.output_extra = output_extra

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Construct full file path
        # Metadata file_path is relative to INPUT_DIR (e.g., "train_images/123.jpg")
        file_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Label
        # We assume label column exists. For test set, it might be a placeholder.
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.output_extra:
            return image, label, row["image_id"]
        else:
            return image, label


def get_dataset(split: str, debug: bool = False, transform=None):
    """
    Factory function to create a CassavaDataset instance based on the split.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, uses a small subset of the data.
        transform (A.Compose, optional): Transforms to apply. If None, uses default for split.

    Returns:
        CassavaDataset: The configured dataset.
    """
    # Select metadata file
    if split == "train":
        csv_path = Config.TRAIN_METADATA
        default_transform_type = "train"
    elif split == "val":
        csv_path = Config.VAL_METADATA
        default_transform_type = "valid"
    elif split == "test":
        csv_path = Config.TEST_METADATA
        default_transform_type = "test"
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load Metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debug Mode: Subset data
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
        # print(f"DEBUG MODE: Loaded {len(df)} samples for {split} split.")

    # Determine Transforms
    if transform is None:
        transform = get_transforms(default_transform_type)

    # For test split, we usually want the image_id for submission
    output_extra = split == "test"

    return CassavaDataset(df, transforms=transform, output_extra=output_extra)

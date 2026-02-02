import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    Responsible for loading images from disk, applying transformations,
    and serving them as tensors with corresponding labels or IDs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms=None,
        input_dir: str = Config.INPUT_DIR,
        sample_size: int = None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            transforms (albumentations.Compose, optional): Transformations to apply.
            input_dir (str): Root directory where images are stored.
            sample_size (int, optional): If provided, limits the dataset to this many samples.
        """
        # Allow subsetting for debugging or quick experiments
        if sample_size is not None and sample_size < len(df):
            self.df = df.iloc[:sample_size].reset_index(drop=True)
        else:
            self.df = df

        self.transforms = transforms
        self.input_dir = input_dir

        # Determine dataset mode based on available columns
        self.has_label = "label" in self.df.columns
        self.has_id = "id" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Resolve full file path
        # Metadata contains relative paths (e.g., "train/cat.0.jpg")
        filepath = os.path.join(self.input_dir, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(filepath)
        if image is None:
            # Although metadata validation ensures existence, we handle load failures safely
            raise FileNotFoundError(f"Failed to load image at: {filepath}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor manually if no transforms provided
            # Transpose (H, W, C) -> (C, H, W) and normalize to [0, 1]
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return appropriate data pair based on dataset type
        if self.has_label:
            # Training/Validation: Return image and ground truth label
            label = row["label"]
            return image, label
        elif self.has_id:
            # Test: Return image and sample ID for submission mapping
            img_id = row["id"]
            return image, img_id
        else:
            # Fallback for unlabeled data without IDs
            return image, -1


def get_dataset(split: str, transforms=None, sample_size: int = None) -> DogCatDataset:
    """
    Factory function to load metadata and create a DogCatDataset for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        transforms (albumentations.Compose, optional): Transformations to apply.
        sample_size (int, optional): Limit dataset size for debugging.

    Returns:
        DogCatDataset: The instantiated dataset object.
    """
    # Map split names to metadata file paths defined in Config
    if split == "train":
        csv_path = Config.TRAIN_CSV
    elif split == "val":
        csv_path = Config.VAL_CSV
    elif split == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    # Verify metadata file existence
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata file not found for split '{split}' at {csv_path}"
        )

    # Load metadata
    df = pd.read_csv(csv_path)

    # Return dataset instance
    return DogCatDataset(df, transforms=transforms, sample_size=sample_size)

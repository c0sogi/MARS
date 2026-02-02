import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config


class AnimalDataset(Dataset):
    """
    Custom Dataset for Animal Classification.
    Reads images from disk based on metadata paths and applies transformations.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (must include 'file_path' and 'Category').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.df.iloc[idx]

        # Construct absolute file path
        # Config.INPUT_DIR is "./input", row['file_path'] is relative like "train_images/xyz.jpg"
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(image_path)

        # Check if image loaded successfully
        if img is None:
            raise FileNotFoundError(f"Failed to load image at path: {image_path}")

        # Convert BGR (OpenCV default) to RGB (PyTorch default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(img)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Get label
        # For test set, this might be a dummy value (0), which is fine as it's ignored during inference
        label = row["Category"]

        return img, torch.tensor(label, dtype=torch.long)


def load_dataset(split: str, transform=None, debug_size: int = None):
    """
    Factory function to load metadata and create an AnimalDataset instance.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): Transform pipeline to apply.
        debug_size (int, optional): Number of samples to use for debugging.
                                    Overrides Config.DEBUG_SAMPLE_SIZE if provided.

    Returns:
        AnimalDataset: The instantiated dataset.
    """
    # Select metadata file based on split
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Load metadata CSV
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Determine effective debug size
    # Priority: Function argument > Config value > None (Full dataset)
    effective_debug_size = (
        debug_size if debug_size is not None else Config.DEBUG_SAMPLE_SIZE
    )

    # Apply subsampling if requested
    if effective_debug_size is not None and effective_debug_size < len(df):
        # Randomly sample to maintain some distribution properties, using fixed seed for reproducibility
        df = df.sample(n=effective_debug_size, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Create and return dataset
    dataset = AnimalDataset(df, transform=transform)
    return dataset

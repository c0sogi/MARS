import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Handles loading of images and structured metadata.
    """

    def __init__(self, metadata_path, transform=None, mode="train", limit=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (callable, optional): Pipeline of transforms to apply to the image.
            mode (str): 'train', 'val', or 'test'. Used to determine if target is returned.
            limit (int, optional): If provided, limits the dataset to the first N samples (for debugging).
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.mode = mode

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Apply limit for debugging if requested
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        # Define the 12 binary feature columns
        self.meta_cols = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative (e.g., "train/{id}.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read with OpenCV
        image = cv2.imread(img_path)

        # Robustness check: if image fails to load, create a blank one
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for compatibility with standard torchvision/HF transforms
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # 2. Extract Metadata Features
        # Select binary columns and convert to float tensor
        meta = row[self.meta_cols].values.astype(np.float32)
        meta = torch.tensor(meta, dtype=torch.float32)

        # 3. Extract Target
        if self.mode in ["train", "val"]:
            target = row["Pawpularity"]
            target = torch.tensor(target, dtype=torch.float32)
        else:
            # For test set, return a dummy value
            target = torch.tensor(0.0, dtype=torch.float32)

        # 4. Extract ID
        sample_id = row["Id"]

        return image, meta, target, sample_id


def get_dataset(split, transform=None, debug=False):
    """
    Factory function to create a PetDataset for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): Transform pipeline.
        debug (bool): If True, limits the dataset size for faster debugging.

    Returns:
        PetDataset: The initialized dataset.
    """
    # Determine limit based on debug flag
    limit = 100 if debug else None

    if split == "train":
        return PetDataset(
            Config.TRAIN_METADATA_PATH, transform=transform, mode="train", limit=limit
        )
    elif split == "val":
        return PetDataset(
            Config.VAL_METADATA_PATH, transform=transform, mode="val", limit=limit
        )
    elif split == "test":
        return PetDataset(
            Config.TEST_METADATA_PATH, transform=transform, mode="test", limit=limit
        )
    else:
        raise ValueError(
            f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
        )

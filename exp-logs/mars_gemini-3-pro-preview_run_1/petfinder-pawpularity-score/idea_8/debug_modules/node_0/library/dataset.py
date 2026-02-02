import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Creates the transformation pipeline for image preprocessing.

    Args:
        img_size (int): Target height and width for resizing.
        mean (tuple): Normalization mean values (RGB).
        std (tuple): Normalization standard deviation values (RGB).

    Returns:
        A.Compose: The composition of transforms including Resize, Normalize, and ToTensorV2.
    """
    return A.Compose(
        [
            # Warping resize (no padding) as per strategy to avoid artifacts
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pawpularity Contest.
    Loads images and corresponding metadata/targets.
    """

    def __init__(self, df, root_dir, transform=None, return_id=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            root_dir (str): Root directory containing the images (e.g., './input').
            transform (callable, optional): Albumentations transform pipeline.
            return_id (bool): Whether to return the sample ID in the output dict.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id

        # List of binary metadata features provided in the dataset
        self.meta_cols = [
            "Focus",
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

        # Check if the target column exists in the dataframe
        self.target_col = "Pawpularity"
        self.has_target = self.target_col in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct the full image path
        # row['file_path'] is relative (e.g., 'train/xxx.jpg')
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing images gracefully
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback: simple conversion to tensor if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Extract binary metadata features as a float tensor
        meta = row[self.meta_cols].values.astype(np.float32)

        # Prepare the output dictionary
        sample = {"image": image, "meta": torch.tensor(meta, dtype=torch.float32)}

        # Add target if available
        if self.has_target:
            target = row[self.target_col]
            # Keep target as float32
            sample["target"] = torch.tensor(target, dtype=torch.float32)

        # Add ID if requested
        if self.return_id:
            sample["id"] = row["Id"]

        return sample


def load_dataset(split="train", debug=None):
    """
    Helper function to load the appropriate metadata DataFrame.

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool, optional): Overrides Config.DEBUG if provided.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if debug is None:
        debug = Config.DEBUG

    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()

    return df

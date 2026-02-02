import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def load_metadata(phase: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for a specific phase (train, val, test).
    Implements caching logic using Parquet files to ensure efficient reloading.

    Args:
        phase (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata containing 'id', 'label', and 'file_path'.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{phase}_metadata.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached {phase} metadata from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. IF loading fails OR load_cached_data is False: Compute/process from scratch.
    if phase == "train":
        source_path = Config.TRAIN_META_PATH
    elif phase == "val":
        source_path = Config.VAL_META_PATH
    elif phase == "test":
        source_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    print(f"Loading {phase} metadata from {source_path}")
    df = pd.read_csv(source_path)

    # Save the result to the cache directory
    try:
        df.to_parquet(cache_path)
        print(f"Saved {phase} metadata to cache at {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return df


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the given phase.

    Args:
        phase (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Define normalization parameters (ImageNet defaults)
    mean = Config.MEAN
    std = Config.STD
    crop_size = Config.CROP_SIZE

    if phase == "train":
        return A.Compose(
            [
                # Crop center 64x64 to capture 32x32 ROI + context
                A.CenterCrop(height=crop_size, width=crop_size),
                # Augmentations: Flips and Rotations are safe for pathology patches
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic cropping and normalization
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology Tumor Detection.
    Handles loading images from disk and applying transformations.
    """

    def __init__(
        self, df: pd.DataFrame, transform=None, data_root: str = Config.INPUT_DIR
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'label', and 'file_path'.
            transform (callable, optional): Albumentations transform pipeline.
            data_root (str): Root directory containing the image files.
        """
        self.df = df
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata for the current sample
        row = self.df.iloc[idx]
        rel_path = row["file_path"]
        label = row["label"]
        img_id = row["id"]

        # Construct full image path
        img_path = os.path.join(self.data_root, rel_path)

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return tuple: (image_tensor, label_tensor, id_string)
        # Label is converted to float32 for BCEWithLogitsLoss compatibility
        return image, torch.tensor(label, dtype=torch.float32), img_id

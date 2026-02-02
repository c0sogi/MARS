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
    Implements caching using parquet to satisfy deterministic processing requirements.

    Args:
        phase (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{phase}_metadata.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {phase} metadata from cache.") # Optional: Silent execution preferred
            return df
        except Exception:
            # If load fails, fall back to processing
            pass

    # Load from source
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

    df = pd.read_csv(source_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to cache metadata: {e}")

    return df


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for a specific phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # All pipelines start with CenterCrop to 64x64 as per Idea
    # Original images are 96x96. We want center 64x64.
    common_transforms = [
        A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
    ]

    if phase == "train":
        # Augmentations for training
        transforms = common_transforms + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    else:
        # Validation and Test (Deterministic)
        transforms = common_transforms + [
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology Tumor Detection.
    Handles loading TIFF images, converting color space, and applying transforms.
    """

    def __init__(self, df: pd.DataFrame, phase: str = "train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'label', and 'file_path'.
            phase (str): Phase of operation ('train', 'val', 'test').
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.df = df
        self.phase = phase
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/xxxx.tif"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Robustness check
        if image is None:
            # Return a blank image or handle error.
            # For this task, we assume data integrity based on metadata checks.
            # Creating a black image of expected size (96x96) to avoid crash.
            image = np.zeros(
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label
        # In test set, label is placeholder 0
        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label

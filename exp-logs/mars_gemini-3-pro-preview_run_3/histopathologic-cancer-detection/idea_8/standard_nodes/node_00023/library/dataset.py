import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    # Standard ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    crop_size = Config.CROP_SIZE

    if phase == "train":
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def load_metadata(phase: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for the specified phase with caching logic.

    Args:
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing absolute file paths and labels.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"cached_{phase}_metadata.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify if loaded data is valid (simple check)
            if "file_path" in df.columns and len(df) > 0:
                print(f"Loaded {phase} metadata from cache: {cache_path}")
                return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    # Load from source CSV
    if phase == "train":
        csv_path = Config.TRAIN_META_PATH
    elif phase == "val":
        csv_path = Config.VAL_META_PATH
    elif phase == "test":
        csv_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Convert relative paths to absolute paths
    # Metadata contains paths like "train/id.tif" or "test/id.tif"
    # We need "./input/train/id.tif"
    df["file_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved {phase} metadata to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Pathology Images.
    """

    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
            transforms (A.Compose, optional): Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["file_path"]

        # Read image
        image = cv2.imread(path)
        if image is None:
            # Handle missing file gracefully, though analysis suggests no missing files
            # Creating a black image of expected size (96x96) to prevent crash
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Get label
        # In test set, label is 0 (placeholder)
        label = row["label"]

        # Return float tensor for label to match BCEWithLogitsLoss expectation usually,
        # or Long for CrossEntropy. Config says NUM_CLASSES=1, implying Binary Classification.
        # Usually BCE takes Float.
        return image, torch.tensor(label, dtype=torch.float32)

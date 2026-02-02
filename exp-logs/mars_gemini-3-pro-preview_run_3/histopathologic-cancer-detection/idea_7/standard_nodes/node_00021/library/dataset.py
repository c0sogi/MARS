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
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    # Base transform: Center Crop to 64x64 as per strategy (32px ROI + 16px context)
    # The original images are 96x96 based on analysis, so we crop center 64.
    transforms_list = [
        A.CenterCrop(height=Config.CENTER_CROP_SIZE, width=Config.CENTER_CROP_SIZE)
    ]

    if phase == "train":
        # Geometric augmentations for rotational invariance
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # Normalization and Tensor conversion
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def load_dataset_metadata(phase: str, load_cached_data: bool = True):
    """
    Loads dataset metadata with caching mechanism.

    Args:
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Determine paths based on phase
    if phase == "train":
        csv_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHED_TRAIN_METADATA
    elif phase == "val":
        # Note: Config doesn't explicitly define CACHED_VAL_METADATA in the provided snippet,
        # but we should follow the pattern. We'll construct it.
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_path = os.path.join(Config.WORKING_DIR, "cached_val_metadata.parquet")
    elif phase == "test":
        csv_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHED_TEST_METADATA
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {phase} metadata from cache: {cache_path}")
            return df
        except Exception:
            # If load fails, fall through to scratch loading
            pass

    # 2. Compute/Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure file_path is treated as string
    df["file_path"] = df["file_path"].astype(str)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {phase} metadata to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology images.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label'.
            transform (albumentations.Compose, optional): Transformations to apply.
        """
        self.df = df
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # Metadata contains relative paths like "train/id.tif"
        full_path = os.path.join(self.input_dir, row["file_path"])

        # Read image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should not happen based on analysis)
            # Return a black image of expected size to prevent crash
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            # (Though get_transforms should usually be used)
            image = ToTensorV2()(image=image)["image"]

        # Get label
        # For test set, label might be placeholder 0, but we still return it.
        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label

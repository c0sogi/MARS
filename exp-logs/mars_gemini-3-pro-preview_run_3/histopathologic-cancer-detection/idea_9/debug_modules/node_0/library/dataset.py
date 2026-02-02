import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_dataset_metadata(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads dataset metadata with a strict caching mechanism using Parquet.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The requested metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_metadata.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {split} metadata from cache: {cache_path}") # Silent execution preferred
            return df
        except Exception:
            # If cache is corrupt, proceed to load from source
            pass

    # 2. Load from source
    if split == "train":
        source_path = Config.TRAIN_META_PATH
    elif split == "val":
        source_path = Config.VAL_META_PATH
    elif split == "test":
        source_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata file not found: {source_path}")

    df = pd.read_csv(source_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {split} metadata to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    return df


def get_transforms(data_split: str) -> A.Compose:
    """
    Returns the Albumentations transformations for a given data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    # Base transforms applied to all splits
    # Center crop to 64x64 as per Idea 9 strategy
    transforms_list = [
        A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
    ]

    # Augmentations for training only
    if data_split == "train":
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # Normalization and Tensor conversion (Standard ImageNet stats)
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology images.
    """

    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' (if available).
            transforms (A.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input directory (e.g., "train/xxx.tif")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing files gracefully by creating a black image
            # This prevents crashing during training if a file is corrupt/missing
            image = np.zeros(
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transforms provided (should not happen in this pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Get label if it exists (test set might have placeholder 0s, but we treat them same structure)
        label = row["label"]

        # Return tuple suitable for DataLoader
        return image, torch.tensor(label, dtype=torch.float32)

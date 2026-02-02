import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_data(csv_path, cache_name, load_cached_data=True):
    """
    Loads the dataset dataframe with strict caching logic using Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name for the cached file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_file = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception:
            # If loading fails, proceed to compute/load from source
            pass

    # 2. Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_file}. Error: {e}")

    return df


def get_transforms(data, img_size):
    """
    Generates the Albumentations transform pipeline based on the strategy.

    Strategy:
    - Strong Geometric Augmentations (ShiftScaleRotate, HorizontalFlip).
    - Strict Exclusion: No VerticalFlip, No Cutout, No Photometric (Brightness/Contrast).

    Args:
        data (str): 'train' or 'valid' (applies to test as well).
        img_size (int): Target image resolution (e.g., 512 for Teacher, 384 for Student).

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentation acting as regularizer
                A.HorizontalFlip(p=Config.AUG_HORIZONTAL_FLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=Config.SHIFT_LIMIT,
                    scale_limit=Config.SCALE_LIMIT,
                    rotate_limit=Config.ROTATE_LIMIT,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_PROB,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading, label extraction, and transformation.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-calculate file paths to avoid overhead in __getitem__
        # Metadata contains relative path in 'file_path' column
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, rel_path)
            for rel_path in df["file_path"].values
        ]

        self.image_ids = df["image_id"].values

        # Extract labels if not in test mode
        if self.mode != "test":
            # Config.CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
            # These columns exist in train.csv/val.csv as 0/1 integers
            self.labels = df[Config.CLASSES].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        path = self.file_paths[idx]
        image = cv2.imread(path)

        if image is None:
            # Fallback for missing images (should not happen given metadata validation)
            # Create a black image of default size to prevent crash
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Return Data
        image_id = self.image_ids[idx]

        if self.mode != "test":
            # Return float labels for BCE/Multi-task loss compatibility
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label, image_id
        else:
            return image, image_id

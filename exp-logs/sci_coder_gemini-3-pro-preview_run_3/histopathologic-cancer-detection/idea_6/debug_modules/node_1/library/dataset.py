import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_dataset_metadata(mode="train", load_cached_data=True):
    """
    Loads dataset metadata with a caching mechanism using Parquet.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"cached_{mode}_metadata.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache load fails, proceed to load from source
            pass

    # 2. Load from source if cache missing or load_cached_data is False
    if mode == "train":
        src_path = Config.TRAIN_META_PATH
    elif mode == "val":
        src_path = Config.VAL_META_PATH
    elif mode == "test":
        src_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source metadata file not found: {src_path}")

    df = pd.read_csv(src_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms = []

    # Center Crop to 64x64 (providing context for the 32x32 center)
    transforms.append(A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE))

    if data == "train":
        # Geometric augmentations for rotational invariance
        transforms.extend(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        )

    # Normalization and Tensor conversion
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for loading pathology images from disk.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label'.
            transforms (A.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path. Metadata has relative path (e.g., "train/id.tif")
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(file_path)

        if image is None:
            # Handle missing/corrupt files by returning a black image
            # Original images are 96x96
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get label (float32 for BCE loss)
        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label

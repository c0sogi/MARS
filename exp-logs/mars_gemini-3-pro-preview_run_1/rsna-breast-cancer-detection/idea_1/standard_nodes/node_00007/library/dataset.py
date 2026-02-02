import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import process_image


def prepare_metadata(load_cached_data=True):
    """
    Loads and processes metadata files. Implements caching to parquet.
    Fills missing values for 'age' and ensures correct data types.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    cache_path_train = os.path.join(Config.CACHE_DIR, "processed_train.parquet")
    cache_path_val = os.path.join(Config.CACHE_DIR, "processed_val.parquet")
    cache_path_test = os.path.join(Config.CACHE_DIR, "processed_test.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_path_train)
            and os.path.exists(cache_path_val)
            and os.path.exists(cache_path_test)
        ):
            try:
                df_train = pd.read_parquet(cache_path_train)
                df_val = pd.read_parquet(cache_path_val)
                df_test = pd.read_parquet(cache_path_test)
                return df_train, df_val, df_test
            except Exception:
                pass  # Fallback to processing

    # 2. Process from scratch
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Impute missing Age
    # Calculate mean from training set
    age_mean = df_train["age"].mean()

    df_train["age"] = df_train["age"].fillna(age_mean)
    df_val["age"] = df_val["age"].fillna(age_mean)
    df_test["age"] = df_test["age"].fillna(age_mean)

    # Ensure implant is present (test set has it, but verify)
    # If implant is missing in any row, fill with 0
    if "implant" in df_train.columns:
        df_train["implant"] = df_train["implant"].fillna(0).astype(int)
    if "implant" in df_val.columns:
        df_val["implant"] = df_val["implant"].fillna(0).astype(int)
    if "implant" in df_test.columns:
        df_test["implant"] = df_test["implant"].fillna(0).astype(int)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_train.to_parquet(cache_path_train, index=False)
    df_val.to_parquet(cache_path_val, index=False)
    df_test.to_parquet(cache_path_test, index=False)

    return df_train, df_val, df_test


class MammographyDataset(Dataset):
    """
    PyTorch Dataset for Mammography Cancer Detection.
    Implements 'Early Fusion': Concatenates image with spatial metadata maps.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Image augmentations.
            mode (str): 'train' or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract columns to avoid overhead in __getitem__
        self.file_paths = self.df["file_path"].values
        self.ages = self.df["age"].values
        self.implants = self.df["implant"].values

        if self.mode == "train":
            self.labels = self.df["cancer"].values
        else:
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load and Process Image
        # Returns float32 array in [0, 1] with shape (H, W)
        img = process_image(self.file_paths[idx])

        # 2. Apply Augmentations
        if self.transforms:
            # Albumentations expects (H, W) or (H, W, C).
            # Our img is (H, W).
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # Ensure image is a tensor (if transforms didn't convert it)
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)

        # If image is 2D (H, W), add channel dim -> (1, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # 3. Prepare Metadata Channels
        h, w = img.shape[1], img.shape[2]

        # Normalize Age: Standard Scaling (Mean ~58.68, Std ~10.03)
        # Cite solution_lesson_node_00002: Spatially broadcasting scalar metadata.
        age_val = (self.ages[idx] - 58.68) / 10.03
        age_channel = torch.full((1, h, w), age_val, dtype=torch.float32)

        # Implant: Binary 0/1
        implant_val = float(self.implants[idx])
        imp_channel = torch.full((1, h, w), implant_val, dtype=torch.float32)

        # 4. Concatenate (Early Fusion)
        # Result shape: (3, H, W)
        input_tensor = torch.cat([img, age_channel, imp_channel], dim=0)

        if self.mode == "train":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return input_tensor, label
        else:
            pred_id = self.prediction_ids[idx]
            return input_tensor, pred_id


def get_dataloaders(load_cached_data=True, sample_size=None):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        sample_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Prepare Metadata
    df_train, df_val, df_test = prepare_metadata(load_cached_data=load_cached_data)

    # Debugging: Subsample
    if sample_size is not None:
        df_train = df_train.head(sample_size)
        df_val = df_val.head(sample_size)
        df_test = df_test.head(sample_size)

    # 2. Define Transforms
    # Note: process_image outputs float32 [0, 1].
    # Albumentations works with this.

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, value=0, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ]
    )

    # No TTA for val/test, just tensor conversion
    val_transforms = A.Compose([ToTensorV2()])

    # 3. Create Datasets
    train_dataset = MammographyDataset(
        df_train, transforms=train_transforms, mode="train"
    )
    val_dataset = MammographyDataset(df_val, transforms=val_transforms, mode="train")
    test_dataset = MammographyDataset(df_test, transforms=val_transforms, mode="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Avoid incomplete batches for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

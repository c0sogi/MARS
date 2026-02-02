import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def worker_init_fn(worker_id):
    np.random.seed(Config.SEED + worker_id)


def _load_processed_df(metadata_path, cache_name, load_cached_data=True):
    """
    Loads the metadata dataframe, resolves full image paths, and handles caching.
    Strictly follows the required caching logic:
    1. If load_cached_data is True, try to load from parquet.
    2. If fails or load_cached_data is False, compute from scratch and save.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to computation
            pass

    # 2. Compute from scratch
    df = pd.read_csv(metadata_path)

    # Resolve full paths
    # Metadata contains paths relative to input root (e.g. "train_images/img.jpg")
    df["full_path"] = df["file_path"].apply(
        lambda x: os.path.join(Config.INPUT_ROOT, x)
    )

    # 3. Save to cache
    df.to_parquet(cache_path)

    return df


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease images.
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]
        label = row["label"]

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Handle case where image might not exist or be corrupt
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_transforms(phase):
    """
    Returns the Albumentations transformations for the specified phase.

    Args:
        phase (str): One of 'train', 'val', 'test'.
    """
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE)),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.CoarseDropout(
                    max_holes=12,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # --- Load DataFrames ---
    train_df = _load_processed_df(Config.TRAIN_METADATA, "train_meta", load_cached_data)
    val_df = _load_processed_df(Config.VAL_METADATA, "val_meta", load_cached_data)
    test_df = _load_processed_df(Config.TEST_METADATA, "test_meta", load_cached_data)

    # --- Handle Debug Mode ---
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # --- Create Datasets ---
    train_dataset = CassavaDataset(train_df, transforms=get_transforms("train"))
    val_dataset = CassavaDataset(val_df, transforms=get_transforms("val"))
    test_dataset = CassavaDataset(test_df, transforms=get_transforms("test"))

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader

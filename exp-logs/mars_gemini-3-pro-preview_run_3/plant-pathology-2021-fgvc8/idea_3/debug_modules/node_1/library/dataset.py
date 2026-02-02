import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Caching & Processing Logic
# ==========================================


def process_metadata(csv_path, cache_filename, load_cached_data=True, debug=False):
    """
    Loads metadata CSV, processes labels into multi-hot format, and handles caching.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_filename (str): Name of the parquet file to save/load.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: Processed dataframe with multi-hot label columns.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if debug:
                df = df.head(100)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Vectorized One-Hot Encoding for space-delimited labels
    # This creates a binary column for every unique token found in 'labels'
    dummies = df["labels"].str.get_dummies(sep=" ")

    # Ensure all expected columns from Config.LABELS exist (handle missing classes in splits)
    for label in Config.LABELS:
        if label not in dummies.columns:
            dummies[label] = 0

    # Select only the expected columns in the correct alphabetical order
    dummies = dummies[Config.LABELS]

    # Concatenate the multi-hot vectors back to the main dataframe
    df = pd.concat([df, dummies], axis=1)

    # Save to cache (save full dataset before debugging crop)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    # If debug, return subset
    if debug:
        df = df.head(100)

    return df


# ==========================================
# Augmentations
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                # Strict lower scale limit of 0.5 as per Idea description
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=(Config.AUG_SCALE_MIN, 1.0),
                    p=1.0,
                ),
                # Rotational invariance
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Mild brightness and contrast perturbations
                A.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=0,
                    hue=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


# ==========================================
# Dataset Class
# ==========================================


class AppleDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file paths and multi-hot labels.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.transform = transform
        self.file_paths = df["file_path"].values

        # Pre-extract labels as a numpy array for faster access during training
        # Shape: (N, Num_Classes)
        self.labels = df[Config.LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images/corrupt files
            # Create a black image to prevent crashing
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get target
        target = self.labels[idx]

        return image, torch.tensor(target)


# ==========================================
# Data Loaders
# ==========================================


def get_loaders(load_cached_data=True, debug=False):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use parquet caching.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 1. Process Metadata
    train_df = process_metadata(
        Config.TRAIN_CSV,
        "train_processed.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    val_df = process_metadata(
        Config.VAL_CSV,
        "val_processed.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    test_df = process_metadata(
        Config.TEST_CSV,
        "test_processed.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Create Datasets
    train_dataset = AppleDataset(train_df, transform=get_transforms(mode="train"))

    val_dataset = AppleDataset(val_df, transform=get_transforms(mode="val"))

    test_dataset = AppleDataset(test_df, transform=get_transforms(mode="test"))

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

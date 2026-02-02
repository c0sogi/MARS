import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transform=None, test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            transform (albumentations.Compose): Transformations to apply.
            test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.transform = transform
        self.test = test

        # Pre-compute full file paths to avoid overhead in __getitem__
        # Metadata contains relative paths like 'images/Train_0.jpg'
        # Config.INPUT_DIR is './input'
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, path) for path in df["file_path"].values
        ]

        if not self.test:
            # Extract labels: healthy, multiple_diseases, rust, scab
            self.labels = df[Config.CLASS_LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (should not happen in pipeline)
            # Just convert to tensor
            image = ToTensorV2()(image=image)["image"]

        if self.test:
            return image, torch.tensor(0)  # Return dummy label for test
        else:
            return image, torch.tensor(self.labels[idx])


def get_transforms(data="train"):
    """
    Returns the Albumentations transformations based on the data split.

    Args:
        data (str): 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                # Resize to ensure consistent input size
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Strong Geometric Augmentations as per strategy
                A.HorizontalFlip(p=Config.RANDOM_FLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=Config.SHIFT_LIMIT,
                    scale_limit=Config.SCALE_LIMIT,
                    rotate_limit=Config.ROTATE_LIMIT,
                    p=Config.SHIFT_SCALE_ROTATE_PROB,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalization (ImageNet mean/std)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                # Resize only for validation/inference
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


def load_data_frames(load_cached_data=True):
    """
    Loads train, val, and test dataframes from metadata or cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = Config.get_cache_path("train_df.parquet")
    val_cache = Config.get_cache_path("val_df.parquet")
    test_cache = Config.get_cache_path("test_df.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        try:
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        except Exception:
            # Fallback to loading from source if cache is corrupt
            pass

    # Load from metadata CSVs
    train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_path = os.path.join(Config.METADATA_DIR, "val.csv")
    test_path = os.path.join(Config.METADATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Metadata file not found: {train_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_df)
               train_df is returned for class weight calculation if needed.
    """
    train_df, val_df, test_df = load_data_frames(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transform=get_transforms(data="train"), test=False
    )

    val_dataset = AppleDataset(
        val_df, transform=get_transforms(data="valid"), test=False
    )

    test_dataset = AppleDataset(
        test_df, transform=get_transforms(data="valid"), test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Normalization stability
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

    return train_loader, val_loader, test_loader, train_df

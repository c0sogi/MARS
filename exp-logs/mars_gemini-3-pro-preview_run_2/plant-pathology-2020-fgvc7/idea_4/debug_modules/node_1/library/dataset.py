import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import worker_init_fn

# Define label columns in the specific order required for submission/training
LABEL_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transform pipeline based on the data type.

    Args:
        data_type (str): One of "train", "valid", or "test".

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # CoarseDropout specifically configured for localized disease spots
                A.CoarseDropout(**Config.COARSE_DROPOUT_PARAMS),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, test_mode=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transforms to apply.
            test_mode (bool): If True, returns (image, image_id).
                              If False, returns (image, label_tensor).
        """
        self.df = df
        self.transforms = transforms
        self.test_mode = test_mode

        # Pre-compute full file paths
        # Metadata file_path is relative, e.g., "images/Train_0.jpg"
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        # Pre-compute labels for training/validation
        if not self.test_mode:
            self.labels = df[LABEL_COLS].values.astype(np.float32)
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.test_mode:
            return image, self.image_ids[idx]
        else:
            label = torch.tensor(self.labels[idx])
            return image, label


def process_and_cache_data(load_cached_data=True):
    """
    Loads metadata CSVs, processes them, and caches them as Parquet files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_cache.parquet")
    val_cache_path = os.path.join(cache_dir, "val_cache.parquet")
    test_cache_path = os.path.join(cache_dir, "test_cache.parquet")

    # Check if cache exists and loading is requested
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
    else:
        # Load from original metadata CSVs
        train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
        val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

        # Save to cache for future runs
        train_df.to_parquet(train_cache_path)
        val_df.to_parquet(val_cache_path)
        test_df.to_parquet(test_cache_path)

    return train_df, val_df, test_df


def get_dataloaders(debug=Config.DEBUG, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for debugging.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_df, val_df, test_df = process_and_cache_data(
        load_cached_data=load_cached_data
    )

    # Subset data if in debug mode
    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Instantiate Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), test_mode=False
    )

    val_dataset = AppleDataset(
        val_df, transforms=get_transforms("valid"), test_mode=False
    )

    test_dataset = AppleDataset(
        test_df, transforms=get_transforms("test"), test_mode=True
    )

    # Create DataLoaders
    # Using worker_init_fn ensures deterministic behavior across workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader

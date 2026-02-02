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


def get_transforms(img_size: int, data_split: str = "train"):
    """
    Returns the Albumentations transform pipeline based on the data split.

    Args:
        img_size (int): The target resolution for the image (e.g., 384, 480).
        data_split (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # CoarseDropout to force distributed feature learning
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
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
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, output_label: bool = True):
        """
        PyTorch Dataset for Apple Disease Detection.

        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (callable, optional): Albumentations transform pipeline.
            output_label (bool): If True, returns (image, target). If False, returns (image, image_id).
        """
        self.df = df
        self.transform = transform
        self.output_label = output_label

        # Pre-construct full file paths
        # df['file_path'] contains relative paths like 'images/Train_0.jpg'
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if self.output_label:
            # Targets: [Is_Rust, Is_Scab]
            self.labels = df[["target_rust", "target_scab"]].values.astype(np.float32)
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(path)
        if image is None:
            # Safety fallback, though data verification passed
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.output_label:
            label = torch.tensor(self.labels[idx])
            return image, label
        else:
            return image, self.image_ids[idx]


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw metadata DataFrame to generate binary targets for multi-label decomposition.

    Mapping:
        Healthy           -> Rust=0, Scab=0
        Rust              -> Rust=1, Scab=0
        Scab              -> Rust=0, Scab=1
        Multiple Diseases -> Rust=1, Scab=1
    """
    # Check if target columns exist (only for train/val)
    if (
        "rust" in df.columns
        and "scab" in df.columns
        and "multiple_diseases" in df.columns
    ):
        df["target_rust"] = df.apply(
            lambda row: (
                1.0 if (row["rust"] == 1 or row["multiple_diseases"] == 1) else 0.0
            ),
            axis=1,
        )
        df["target_scab"] = df.apply(
            lambda row: (
                1.0 if (row["scab"] == 1 or row["multiple_diseases"] == 1) else 0.0
            ),
            axis=1,
        )
    return df


def load_data(load_cached_data: bool = True):
    """
    Loads training, validation, and test dataframes.
    Implements caching logic using Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists for cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(Config.WORKING_DIR, "train_cache.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cache.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    def get_df(metadata_path, cache_path, is_train=False):
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                pass  # Cache might be corrupt, proceed to process

        # 2. Process from scratch
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file missing: {metadata_path}")

        df = pd.read_csv(metadata_path)

        if is_train:
            df = process_dataframe(df)

        # 3. Save to cache
        df.to_parquet(cache_path)
        return df

    # Load datasets
    train_df = get_df(Config.TRAIN_METADATA_PATH, train_cache, is_train=True)
    val_df = get_df(Config.VAL_METADATA_PATH, val_cache, is_train=True)
    test_df = get_df(Config.TEST_METADATA_PATH, test_cache, is_train=False)

    # Debug mode: subsample data
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    return train_df, val_df, test_df


def get_loaders(train_df, val_df, img_size, batch_size, num_workers=Config.NUM_WORKERS):
    """
    Constructs DataLoaders for training and validation.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_ds = AppleDataset(
        train_df, transform=get_transforms(img_size, "train"), output_label=True
    )

    val_ds = AppleDataset(
        val_df, transform=get_transforms(img_size, "val"), output_label=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(test_df, img_size, batch_size, num_workers=Config.NUM_WORKERS):
    """
    Constructs DataLoader for testing/inference.

    Args:
        test_df (pd.DataFrame): Test data.
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: The test data loader.
    """
    test_ds = AppleDataset(
        test_df, transform=get_transforms(img_size, "test"), output_label=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

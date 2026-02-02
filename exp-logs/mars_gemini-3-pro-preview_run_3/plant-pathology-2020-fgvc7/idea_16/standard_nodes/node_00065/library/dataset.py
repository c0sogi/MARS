import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Classification.
    Handles image loading, augmentation, and label extraction.
    """

    def __init__(self, df, transforms=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, labels).
            transforms (albumentations.Compose): Augmentation pipeline.
            output_label (bool): Whether to return labels (True for Train/Val, False for Test).
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label
        self.labels = Config.LABELS

        # Pre-compute full file paths
        # Metadata file_path is relative to input_dir (e.g., "images/Train_0.jpg")
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if self.output_label:
            # Extract one-hot encoded labels as float array
            # Columns are guaranteed to exist by metadata generation
            self.y = df[self.labels].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        path = self.file_paths[idx]
        img = cv2.imread(path)

        # Safety check
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # Return image and label (if requested)
        if self.output_label:
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, target
        else:
            return img


def get_transforms(data_type):
    """
    Returns the augmentation pipeline based on the data type.

    Strategy:
    - Train: Strong Geometric Augmentations (Rotational Invariance) to handle arbitrary leaf orientation.
             No photometric distortions (brightness/contrast) to preserve disease color signals.
    - Valid/Test: Resize and Normalize only.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                # Rotational Invariance Strategy
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                # Standard Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data_type in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def load_data(load_cached_data=True):
    """
    Loads train, val, and test dataframes.
    Implements caching using parquet files in Config.WORKING_DIR to speed up subsequent runs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_df.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_df.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_df.parquet")

    # Attempt to load from cache
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
            # If cache loading fails, proceed to load from source
            pass

    # Load from metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    Handles debug sampling if Config.DEBUG is True.
    """
    # Load DataFrames
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Apply Debug Sampling
    if Config.DEBUG:
        train_df = train_df.sample(
            min(len(train_df), Config.DEBUG_SAMPLE_SIZE)
        ).reset_index(drop=True)
        val_df = val_df.sample(min(len(val_df), Config.DEBUG_SAMPLE_SIZE)).reset_index(
            drop=True
        )
        test_df = test_df.sample(
            min(len(test_df), Config.DEBUG_SAMPLE_SIZE)
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), output_label=True
    )

    val_dataset = AppleDataset(
        val_df, transforms=get_transforms("valid"), output_label=True
    )

    test_dataset = AppleDataset(
        test_df,
        transforms=get_transforms("test"),
        output_label=False,  # Test set has no ground truth labels
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for BatchNorm stability
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

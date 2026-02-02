import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from library.config import Config
from library.augmentations import get_train_transforms, get_valid_transforms


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading via OpenCV and soft-target label extraction.
    """

    def __init__(self, df, transforms=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing image paths and labels.
            transforms (albumentations.Compose): Augmentation pipeline.
            is_test (bool): If True, returns image_id instead of labels.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_test = is_test
        self.target_cols = Config.CLASS_NAMES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative to './input' (e.g., 'images/Train_0.jpg')
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though verification ensures they exist)
            # Create a black image of the expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            return image, row["image_id"]
        else:
            # Extract soft targets as float32 tensor
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)


def get_data_loaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    split_seed=None,
):
    """
    Creates training and validation DataLoaders.

    Supports the 'Stratified Shuffle-Split' strategy:
    - If split_seed is provided, merges default train/val sets and re-splits
      dynamically to create unique folds for the ensemble.
    - If split_seed is None, uses the fixed metadata splits.

    Args:
        train_metadata_path (str): Path to training metadata CSV.
        val_metadata_path (str): Path to validation metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        split_seed (int, optional): Seed for dynamic re-splitting.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load default metadata
    df_train_meta = pd.read_csv(train_metadata_path)
    df_val_meta = pd.read_csv(val_metadata_path)

    if split_seed is not None:
        # Dynamic Stratified Shuffle-Split Strategy
        # Combine datasets
        df_full = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

        # Perform stratified split
        # We use 'stratify_label' which represents the dominant class
        train_df, val_df = train_test_split(
            df_full,
            test_size=0.20,
            stratify=df_full["stratify_label"],
            random_state=split_seed,
            shuffle=True,
        )
    else:
        # Use fixed splits from metadata
        train_df = df_train_meta
        val_df = df_val_meta

    # Debug Mode: Subsample data
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), 64), random_state=42
        ).reset_index(drop=True)
        val_df = val_df.sample(n=min(len(val_df), 32), random_state=42).reset_index(
            drop=True
        )

    # Create Datasets
    train_dataset = AppleDataset(
        df=train_df, transforms=get_train_transforms(), is_test=False
    )

    val_dataset = AppleDataset(
        df=val_df, transforms=get_valid_transforms(), is_test=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_metadata_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates the test DataLoader for inference.

    Args:
        test_metadata_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: Test data loader.
    """
    df_test = pd.read_csv(test_metadata_path)

    test_dataset = AppleDataset(
        df=df_test, transforms=get_valid_transforms(), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config


class PetDataset(Dataset):
    """
    Dataset class for loading Dog vs Cat images.
    Handles both labeled data (train/val) and unlabeled data (test).
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            transforms (A.Compose): Albumentations transforms to apply.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata paths are relative to INPUT_DIR
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(filepath)
        if image is None:
            # Fallback or error handling; usually dataset is clean per validation
            raise FileNotFoundError(f"Image not found at {filepath}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return format depends on mode
        if self.mode == "test":
            # For test, we need the ID to create the submission file
            return image, row["id"]
        else:
            # For train/val, we return the label
            # Ensure label is float for BCE/LogLoss (though CrossEntropy uses long,
            # binary tasks often use BCEWithLogits which expects float)
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=Config.IMAGE_SIZE,
                    width=Config.IMAGE_SIZE,
                    scale=(Config.CROP_SCALE_MIN, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize to target size deterministically
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_folds(load_cached_data=True):
    """
    Loads metadata, combines train/val splits, and generates stratified folds.
    Caches the result to disk to ensure reproducibility and speed.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing all training data with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch if cache missing or forced reload
    # Load original metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Concatenate to form the full dataset
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Initialize fold column
    full_df["fold"] = -1

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    for fold_idx, (_, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold_idx

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path)

    return full_df


def get_train_val_loaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index to use for validation (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached fold splits.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data with fold info
    df = prepare_folds(load_cached_data=load_cached_data)

    # Split into train and validation
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Debugging: Subsample if configured
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = PetDataset(
        train_df, transforms=get_transforms(mode="train"), mode="train"
    )
    val_dataset = PetDataset(val_df, transforms=get_transforms(mode="val"), mode="val")

    # Create DataLoaders
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

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: The test data loader.
    """
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    test_dataset = PetDataset(
        test_df, transforms=get_transforms(mode="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

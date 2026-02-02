import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.target_cols = Config.TARGET_COLS
        self.file_paths = df["file_path"].values

        if not self.is_test:
            self.labels = df[self.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # Metadata file_path is relative to input dir (e.g., "images/Train_0.jpg")
        file_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Read image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle targets
        if self.is_test:
            # Return dummy target for test set
            target = torch.zeros(len(self.target_cols), dtype=torch.float32)
        else:
            target = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, target


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data_type (str): 'train' or 'valid'.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),  # Explicitly included per strategy
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test transforms
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def _get_folded_dataframe(load_cached_data=True):
    """
    Loads metadata, combines train/val splits, and generates/caches Stratified K-Folds.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Proceed to regeneration if load fails

    # 2. Compute from scratch
    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Concatenate to form the full available training set
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Initialize fold column
    df_full["fold"] = -1

    # Perform Stratified K-Fold
    # We use 'stratify_label' which was generated in the metadata script
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # If stratify_label is missing (unlikely), reconstruct it
    if "stratify_label" not in df_full.columns:
        df_full["stratify_label"] = df_full[Config.TARGET_COLS].idxmax(axis=1)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["stratify_label"])
    ):
        df_full.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    try:
        df_full.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df_full


def get_loaders(fold=0, mode="calibration"):
    """
    Creates DataLoaders for training and validation.

    Args:
        fold (int): The fold index to use for validation (0 to N_FOLDS-1).
        mode (str): 'calibration' (returns train/val split) or 'production' (returns full train).

    Returns:
        tuple: (train_loader, val_loader)
               val_loader is None if mode == 'production'.
    """
    # Get dataframe with fold assignments
    df = _get_folded_dataframe(load_cached_data=True)

    if mode == "calibration":
        # Split based on fold
        df_train = df[df["fold"] != fold].reset_index(drop=True)
        df_val = df[df["fold"] == fold].reset_index(drop=True)

        # Create Datasets
        train_dataset = AppleDataset(df_train, transform=get_transforms("train"))
        val_dataset = AppleDataset(df_val, transform=get_transforms("valid"))

        # Create Loaders
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

    elif mode == "production":
        # Use 100% of data for training
        train_dataset = AppleDataset(df, transform=get_transforms("train"))

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        return train_loader, None

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: The test data loader.
    """
    df_test = pd.read_csv(Config.TEST_METADATA)

    test_dataset = AppleDataset(
        df_test, transform=get_transforms("valid"), is_test=True
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

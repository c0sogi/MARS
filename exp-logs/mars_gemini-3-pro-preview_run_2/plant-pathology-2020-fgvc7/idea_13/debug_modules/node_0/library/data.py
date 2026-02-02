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
from library.utils import get_binary_targets, seed_everything


def get_transforms(img_size, mode="train"):
    """
    Generates the Albumentations transformation pipeline.

    Args:
        img_size (int): The input resolution (e.g., 384 or 480).
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # CoarseDropout to force distributed feature learning
                A.CoarseDropout(
                    max_holes=Config.AUG_COARSE_DROPOUT_MAX_HOLES,
                    max_height=Config.AUG_COARSE_DROPOUT_MAX_HEIGHT,
                    max_width=Config.AUG_COARSE_DROPOUT_MAX_WIDTH,
                    min_holes=Config.AUG_COARSE_DROPOUT_MIN_HOLES,
                    min_height=Config.AUG_COARSE_DROPOUT_MIN_HEIGHT,
                    min_width=Config.AUG_COARSE_DROPOUT_MIN_WIDTH,
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
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode

        # Pre-calculate targets for training/validation
        if self.mode in ["train", "val"]:
            # Use the utility function to get the decomposed binary targets (Rust, Scab)
            self.targets = get_binary_targets(self.df)
            self.labels = self.df["stratify_label"].values
        else:
            self.targets = None

        self.image_ids = self.df["image_id"].values
        self.file_paths = self.df["file_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, target
        else:
            image_id = self.image_ids[idx]
            return image, image_id


def get_folds_data(load_cached_data=True):
    """
    Loads or creates the 5-fold cross-validation split data.
    Strictly follows the caching mechanism requirement.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to computation

    # 2. Compute from scratch
    # Load original train and val metadata provided by the system
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine them to perform a fresh 5-fold split
    full_df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Initialize fold column
    full_df["fold"] = -1

    # Assign folds based on the 'stratify_label'
    for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["stratify_label"])):
        full_df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)

    return full_df


def get_train_val_loaders(fold_idx, img_size, batch_size):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1) to use for validation.
        img_size (int): Image resolution.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data with caching
    df = get_folds_data(load_cached_data=True)

    # Split into train and val based on fold index
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Handle DEBUG mode
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transform=get_transforms(img_size, mode="train"), mode="train"
    )
    val_dataset = AppleDataset(
        val_df, transform=get_transforms(img_size, mode="val"), mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(img_size, batch_size):
    """
    Creates DataLoader for the test set.

    Args:
        img_size (int): Image resolution.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test data loader.
    """
    # Load test metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create Dataset
    test_dataset = AppleDataset(
        test_df, transform=get_transforms(img_size, mode="test"), mode="test"
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

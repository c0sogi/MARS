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
from library.utils import seed_everything


class AppleDataset(Dataset):
    """
    Dataset class for Apple Disease Detection.
    Handles loading images, applying augmentations, and returning labels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transforms (albumentations.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.file_paths = df["file_path"].values

        # Labels are only available for train/val
        if self.mode != "test":
            # Ensure we get the columns in the order defined in Config
            self.labels = df[Config.CLASS_LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        file_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            # Return image and dummy label for test
            return image, torch.tensor(0)
        else:
            label = torch.tensor(self.labels[idx])
            return image, label


def get_transforms(img_size, data_split="train"):
    """
    Returns albumentations transforms for train/val/test.

    Args:
        img_size (int): Target image size (height and width).
        data_split (str): 'train', 'val', or 'test'.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentations
                # ShiftScaleRotate with wide limits
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                # Random Flips
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Transpose (strictly maintained for training)
                A.Transpose(p=0.5),
                # EXCLUSIONS:
                # - No Cutout / CoarseDropout (Spatial Occlusion)
                # - No Brightness / Contrast (Photometric)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test: Deterministic Resize and Normalize
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


def prepare_folds(load_cached_data=True):
    """
    Merges train and val metadata, creates stratified folds, and caches the result.

    Args:
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame with 'fold' column.
    """
    folds_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(folds_path):
        try:
            df = pd.read_parquet(folds_path)
            return df
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute folds
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Concatenate to form full training set for CV
    df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Use 'stratify_label' if available, else derive it
    if "stratify_label" in df.columns:
        y = df["stratify_label"]
    else:
        y = df[Config.CLASS_LABELS].idxmax(axis=1)

    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(folds_path)

    return df


def get_loaders(fold, img_size, batch_size, debug=False):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold (int): The fold index to use for validation (0 to N_FOLDS-1).
        img_size (int): Image size for resizing.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Get data with folds
    df = prepare_folds(load_cached_data=True)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Note: In debug mode, stratification might be broken, but that's acceptable for testing pipeline

    # Split train/val
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms(img_size, data_split="train"), mode="train"
    )
    val_dataset = AppleDataset(
        val_df, transforms=get_transforms(img_size, data_split="val"), mode="val"
    )

    # Create Loaders
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
        img_size (int): Image size.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test data loader.
    """
    df = pd.read_csv(Config.TEST_CSV)

    dataset = AppleDataset(
        df, transforms=get_transforms(img_size, data_split="test"), mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader

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
from library.utils import seed_everything


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, output_label=True):
        self.df = df
        self.transforms = transforms
        self.output_label = output_label
        self.root_dir = Config.INPUT_ROOT

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Construct full image path
        # Metadata contains relative path in 'file_path' column
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Handle potential missing/corrupt files gracefully
        if img is None:
            # Return a black image of correct size to prevent crash
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            img = self.transforms(image=img)["image"]
        else:
            # Default transform if none provided
            t = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            img = t(image=img)["image"]

        # Return data based on mode
        if self.output_label:
            label = row["label"]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # For test/inference, return image_id to map predictions
            return img, row["image_id"]


def get_transforms(phase):
    """
    Returns the albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'valid' (also used for test)
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
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
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def process_data(load_cached_data=True):
    """
    Loads training metadata and assigns stratified folds.
    Implements caching mechanism using parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with 'fold' column.
    """
    cache_path = os.path.join(
        Config.OUTPUT_DIR, f"train_with_{Config.N_FOLDS}folds.parquet"
    )

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to re-compute
            pass

    # 2. Compute data from scratch
    # Load original metadata
    df = pd.read_csv(Config.TRAIN_METADATA)

    # Create folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(fold_id=0, load_cached_data=True):
    """
    Creates DataLoaders for the specified fold.

    Args:
        fold_id (int): The fold index to use for validation (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached fold assignments.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data with folds
    df = process_data(load_cached_data=load_cached_data)

    # Split into train and val based on fold_id
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    val_df = df[df["fold"] == fold_id].reset_index(drop=True)

    # Create Datasets
    train_dataset = CassavaDataset(
        train_df, transforms=get_transforms("train"), output_label=True
    )
    val_dataset = CassavaDataset(
        val_df, transforms=get_transforms("valid"), output_label=True
    )

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


def get_test_dataloader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    df_test = pd.read_csv(Config.TEST_METADATA)

    test_dataset = CassavaDataset(
        df_test, transforms=get_transforms("valid"), output_label=False
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

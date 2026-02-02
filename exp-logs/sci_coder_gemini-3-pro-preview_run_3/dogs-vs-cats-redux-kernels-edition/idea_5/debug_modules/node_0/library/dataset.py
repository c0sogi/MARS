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


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.RandomResizedCrop(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.333),
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_train_data(load_cached_data=True):
    """
    Loads training metadata. Merges train and val CSVs to allow for custom Cross-Validation splitting.
    Implements caching using parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "full_train_data.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute/Process data
    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine for CV
    full_df = pd.concat([df_train, df_val], ignore_index=True)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    full_df.to_parquet(cache_path)

    return full_df


def load_test_data(load_cached_data=True):
    """
    Loads test metadata with caching.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_data.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass

    # 2. Compute/Process data
    df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)

    return df


class CatDogDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = row["label"]
            # Return float label for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # For test, return image and ID
            img_id = row["id"]
            return image, torch.tensor(img_id, dtype=torch.long)


def get_dataloaders(fold_idx=None, debug=False, batch_size=Config.BATCH_SIZE):
    """
    Creates training and validation DataLoaders.

    Args:
        fold_idx (int, optional): The fold index (0 to N_FOLDS-1) for validation.
                                  If None, uses the default metadata split.
        debug (bool): If True, uses a small subset of data.
        batch_size (int): Batch size.
    """
    # Load combined data
    # We use load_cached_data=True by default as per requirement logic
    df = load_train_data(load_cached_data=True)

    if debug:
        df = df.sample(n=200, random_state=Config.SEED).reset_index(drop=True)

    if fold_idx is not None:
        # Perform Stratified K-Fold Split
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Create a temporary fold column
        df["fold"] = -1
        for i, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            df.loc[val_idx, "fold"] = i

        train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
        val_df = df[df["fold"] == fold_idx].reset_index(drop=True)
    else:
        # Fallback to original metadata split if no fold specified
        # Reloading from source to ensure exact match with metadata files
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        if debug:
            train_df = train_df.sample(100, random_state=Config.SEED)
            val_df = val_df.sample(100, random_state=Config.SEED)

    # Create Datasets
    train_ds = CatDogDataset(train_df, transforms=get_transforms("train"), mode="train")
    val_ds = CatDogDataset(val_df, transforms=get_transforms("val"), mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(debug=False, batch_size=Config.BATCH_SIZE):
    """
    Creates the test DataLoader.
    """
    df = load_test_data(load_cached_data=True)

    if debug:
        df = df.iloc[:100]

    test_ds = CatDogDataset(df, transforms=get_transforms("test"), mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

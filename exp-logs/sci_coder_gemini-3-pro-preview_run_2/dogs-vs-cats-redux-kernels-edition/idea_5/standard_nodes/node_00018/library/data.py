import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed


def get_transforms(mode="train"):
    """
    Returns the image transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # RandomResizedCrop with scale 0.8 as per strategy
                transforms.RandomResizedCrop(
                    Config.image_size, scale=(Config.min_crop_scale, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Deterministic resize for validation and test
        return transforms.Compose(
            [
                transforms.Resize((Config.image_size, Config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class CatDogDataset(Dataset):
    """
    Dataset class for Dog vs Cat classification.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.input_dir, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(filepath)
        if image is None:
            # In a real scenario we might skip, but here we expect clean data
            raise FileNotFoundError(f"Image not found at {filepath}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms compatibility
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # For test set, return image and ID
            return image, row["id"]
        else:
            # For train/val, return image and label
            # Label is float32 for BCEWithLogitsLoss compatibility
            return image, torch.tensor(row["label"], dtype=torch.float32)


def get_folded_data(load_cached_data=True):
    """
    Loads training data and splits it into folds.
    Implements caching using parquet to ensure deterministic splits.
    """
    cache_path = os.path.join(Config.working_dir, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    print("Generating new folds...")

    # Load provided metadata
    train_df = pd.read_csv(Config.train_metadata)
    val_df = pd.read_csv(Config.val_metadata)

    # Combine into a single dataframe for 5-fold CV
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    full_df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.working_dir, exist_ok=True)
    full_df.to_parquet(cache_path)

    return full_df


def get_train_val_loaders(fold_idx, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for a specific fold.
    """
    if batch_size is None:
        batch_size = Config.batch_size
    if num_workers is None:
        num_workers = Config.num_workers

    # Get data with folds
    df = get_folded_data(load_cached_data=True)

    # Split based on fold index
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = CatDogDataset(
        train_df, transform=get_transforms(mode="train"), mode="train"
    )
    val_dataset = CatDogDataset(
        val_df, transform=get_transforms(mode="val"), mode="val"
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


def get_test_loader(batch_size=None, num_workers=None):
    """
    Creates DataLoader for the test set.
    """
    if batch_size is None:
        batch_size = Config.batch_size
    if num_workers is None:
        num_workers = Config.num_workers

    # Load test metadata
    test_df = pd.read_csv(Config.test_metadata)

    # Create Dataset
    test_dataset = CatDogDataset(
        test_df, transform=get_transforms(mode="test"), mode="test"
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

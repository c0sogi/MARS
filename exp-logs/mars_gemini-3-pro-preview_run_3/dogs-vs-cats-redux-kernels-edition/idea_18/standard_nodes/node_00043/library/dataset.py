import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SEED,
    NUM_WORKERS,
)


class CatDogDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    Handles loading images via OpenCV, converting to RGB, and applying transforms.
    """

    def __init__(self, df, transform=None, data_dir=INPUT_DIR, mode="train"):
        self.df = df
        self.transform = transform
        self.data_dir = data_dir
        self.mode = mode

        # Pre-compute paths to avoid overhead in __getitem__
        self.filepaths = self.df["filepath"].values

        if self.mode != "test":
            self.labels = self.df["label"].values
        else:
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.filepaths[idx]
        img_path = os.path.join(self.data_dir, rel_path)

        # Load image using cv2
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode != "test":
            # Return float label for BCEWithLogitsLoss
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Return ID for submission file creation
            id_val = self.ids[idx]
            return image, id_val


def get_transforms(image_size, mode="train"):
    """
    Returns the appropriate transforms for the given mode and image size.
    Implements the augmentation strategy defined in the idea.
    """
    # Standard ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # Context-Preserving Augmentation: Scale 0.8-1.0 ensures subject is kept
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Color Jitter for lighting invariance
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Valid or Test: Deterministic resizing
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def load_data_and_create_folds(n_folds=5, load_cached_data=True):
    """
    Loads train and validation metadata, combines them, and creates stratified folds.
    Uses caching to store/retrieve the dataframe with fold assignments.
    """
    cache_path = os.path.join(WORKING_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing folds.")

    # 2. Compute from scratch if cache missing or invalid
    print("Creating folds from scratch...")

    # Load metadata provided by the environment
    if not os.path.exists(TRAIN_META_PATH) or not os.path.exists(VAL_META_PATH):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation ran successfully."
        )

    train_df = pd.read_csv(TRAIN_META_PATH)
    val_df = pd.read_csv(VAL_META_PATH)

    # Combine to utilize 100% of data for Cross-Validation
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Create Stratified Folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    full_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return full_df


def get_dataloaders(df, fold, image_size, batch_size, num_workers=NUM_WORKERS):
    """
    Creates train and validation DataLoaders for a specific fold.
    """
    # Split dataframe based on fold column
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = CatDogDataset(
        train_df, transform=get_transforms(image_size, mode="train"), mode="train"
    )

    val_dataset = CatDogDataset(
        val_df, transform=get_transforms(image_size, mode="val"), mode="val"
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
    )

    return train_loader, val_loader


def get_test_loader(image_size, batch_size, num_workers=NUM_WORKERS):
    """
    Creates the test DataLoader using the test metadata.
    """
    if not os.path.exists(TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_META_PATH}")

    test_df = pd.read_csv(TEST_META_PATH)

    test_dataset = CatDogDataset(
        test_df, transform=get_transforms(image_size, mode="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader

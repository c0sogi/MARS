import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Define feature columns based on EDA
# 'Subject Focus' is sometimes labeled 'Focus' in different versions of the dataset
DENSE_FEATURES = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
DENSE_FEATURES_ALT = [
    "Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


class PawpularityDataset(Dataset):
    """
    Dataset class for Pet Pawpularity Prediction.
    Loads images, applies transforms, and returns image, metadata, and target.
    """

    def __init__(self, df, image_dir, transforms=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_test = is_test

        # Determine which column names are present in the dataframe
        if "Subject Focus" in df.columns:
            self.dense_features = DENSE_FEATURES
        else:
            self.dense_features = DENSE_FEATURES_ALT

        # Pre-extract paths and data to avoid overhead in __getitem__
        self.image_paths = df["file_path"].values
        self.meta_data = df[self.dense_features].values.astype(np.float32)

        if not self.is_test:
            self.targets = df[Config.target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # Metadata contains relative paths (e.g., "train/{id}.jpg")
        img_path = os.path.join(self.image_dir, self.image_paths[idx])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for safety, though paths should be verified
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get metadata features
        meta = self.meta_data[idx]

        # Get target
        if self.is_test:
            target = 0.0  # Dummy value for test set
        else:
            # Scale target from [1, 100] to [0, 1] for Sigmoid
            target = self.targets[idx] / 100.0

        return image, torch.tensor(meta), torch.tensor(target)


def get_transforms(mode="train", image_size=224):
    """
    Returns Albumentations transforms for training or validation.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
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
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(fold_idx=0):
    """
    Creates training and validation DataLoaders for a specific fold.
    Merges metadata train/val splits and re-splits using StratifiedKFold.
    """
    # Load existing metadata files
    train_df = pd.read_csv(Config.train_csv_path)
    val_df = pd.read_csv(Config.val_csv_path)

    # Combine them to perform fresh K-Fold splitting
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # Handle Debug Mode
    if Config.debug:
        full_df = full_df.sample(
            n=min(len(full_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # Create bins for stratification based on target variable
    num_bins = int(np.floor(1 + np.log2(len(full_df))))
    full_df["bins"] = pd.cut(full_df[Config.target_col], bins=num_bins, labels=False)

    # Create Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.num_folds, shuffle=True, random_state=Config.seed
    )

    # Get indices for the requested fold
    splits = list(skf.split(full_df, full_df["bins"]))
    train_idx, val_idx = splits[fold_idx]

    train_fold_df = full_df.iloc[train_idx].copy()
    val_fold_df = full_df.iloc[val_idx].copy()

    # Create Datasets
    train_ds = PawpularityDataset(
        train_fold_df,
        Config.input_dir,
        transforms=get_transforms("train", Config.image_size),
        is_test=False,
    )

    val_ds = PawpularityDataset(
        val_fold_df,
        Config.input_dir,
        transforms=get_transforms("valid", Config.image_size),
        is_test=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates a DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.test_csv_path)

    test_ds = PawpularityDataset(
        test_df,
        Config.input_dir,
        transforms=get_transforms("valid", Config.image_size),
        is_test=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import CFG
from library.utils import seed_everything


class CatDogDataset(Dataset):
    """
    Dataset class for loading Dog vs Cat images.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode
        # Pre-compute full file paths
        self.file_paths = [
            os.path.join(CFG.input_dir, fp) for fp in df["filepath"].values
        ]

        if self.mode in ["train", "val"]:
            self.labels = df["label"].values
        else:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Handle potential read errors (though metadata validation passed)
        if image is None:
            # Return a black image of correct size to avoid crashing
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            # Return image and label (float for BCE loss)
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return image, label
        else:
            # Return image and ID for submission mapping
            img_id = self.ids[idx]
            return image, img_id


def get_transforms(data, cfg):
    """
    Returns Albumentations transform pipelines.
    """
    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=cfg.image_size, width=cfg.image_size, scale=cfg.crop_scale
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(height=cfg.image_size, width=cfg.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_loaders(fold, cfg):
    """
    Creates train and validation loaders for a specific fold using Stratified K-Fold.
    Merges train.csv and val.csv from metadata to utilize the full dataset.
    """
    # Load metadata
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)

    # Combine to form full dataset for CV
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Create folds
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    # Iterate to find the indices for the requested fold
    train_fold_df = None
    val_fold_df = None

    for i, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        if i == fold:
            train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)
            break

    if train_fold_df is None or val_fold_df is None:
        raise ValueError(f"Invalid fold {fold} for {cfg.num_folds} splits.")

    # Debug option to run quickly
    if cfg.debug:
        train_fold_df = train_fold_df.sample(n=200, random_state=cfg.seed).reset_index(
            drop=True
        )
        val_fold_df = val_fold_df.sample(n=100, random_state=cfg.seed).reset_index(
            drop=True
        )

    # Transforms
    train_transform = get_transforms("train", cfg)
    val_transform = get_transforms("valid", cfg)

    # Datasets
    train_dataset = CatDogDataset(
        train_fold_df, transform=train_transform, mode="train"
    )
    val_dataset = CatDogDataset(val_fold_df, transform=val_transform, mode="val")

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(cfg):
    """
    Creates the test data loader.
    """
    test_df = pd.read_csv(cfg.test_csv)

    if cfg.debug:
        test_df = test_df.sample(n=100, random_state=cfg.seed).reset_index(drop=True)

    test_transform = get_transforms("test", cfg)
    test_dataset = CatDogDataset(test_df, transform=test_transform, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader

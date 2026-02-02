import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Pre-compute full paths to avoid doing it in __getitem__
        # CFG.input_root is "./input", file_path in metadata is like "train_images/xyz.jpg"
        self.file_paths = [
            os.path.join(CFG.input_root, fp) for fp in df["file_path"].values
        ]
        self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Handle cases where image might not load (though metadata check passed)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label (as long tensor)
        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(data):
    """
    Returns the augmentation pipeline based on the data split.
    Implements the strategy from Idea 7:
    - Resize to 384x384
    - Train: Transpose, HFlip, VFlip (D4 symmetry)
    - Val/Test: Deterministic Resize
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
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
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


def get_loaders():
    """
    Prepares DataLoaders for train, validation, and test sets.
    Reads metadata from ./metadata directory.
    """
    # Load Metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Debug Mode: Subsample data
    if CFG.debug:
        train_df = train_df.sample(
            n=min(len(train_df), 100), random_state=CFG.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 50), random_state=CFG.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), 50), random_state=CFG.seed
        ).reset_index(drop=True)

    # Initialize Datasets
    train_dataset = CassavaDataset(train_df, transform=get_transforms("train"))
    val_dataset = CassavaDataset(val_df, transform=get_transforms("valid"))
    test_dataset = CassavaDataset(test_df, transform=get_transforms("test"))

    # Initialize DataLoaders
    # Drop last for train to maintain batch statistics stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

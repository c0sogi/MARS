import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


def get_transforms(mode="train"):
    """
    Returns the albumentations transforms based on the mode.
    Implements Resolution Discrepancy:
    - Train: RandomResizedCrop to CFG.train_size (256)
    - Val/Test: Resize to CFG.test_size (384) (SmallestMaxSize) then CenterCrop
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(CFG.train_size, CFG.train_size)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # For val and test
        return A.Compose(
            [
                # Resize shortest edge to test_size, maintaining aspect ratio
                A.SmallestMaxSize(max_size=CFG.test_size),
                # Center crop to the square
                A.CenterCrop(height=CFG.test_size, width=CFG.test_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


class INatDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        self.df = df
        self.mode = mode
        self.transform = transform

        # Ensure file paths are correct relative to input root
        self.file_paths = self.df["file_name"].values

        if self.mode in ["train", "val"]:
            self.labels = self.df["category_id"].values
        else:
            self.ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(CFG.input_root, self.file_paths[idx])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for potentially missing images (safety check)
            # Create a black image of training size
            image = np.zeros((CFG.train_size, CFG.train_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            image_id = self.ids[idx]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_loaders():
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(CFG.train_metadata_path)
    val_df = pd.read_csv(CFG.val_metadata_path)
    test_df = pd.read_csv(CFG.test_metadata_path)

    # Debug Mode
    if CFG.debug:
        train_df = train_df.sample(
            n=min(len(train_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)

    # Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Datasets
    train_dataset = INatDataset(train_df, mode="train", transform=train_transform)
    val_dataset = INatDataset(val_df, mode="val", transform=val_transform)
    test_dataset = INatDataset(test_df, mode="test", transform=test_transform)

    # Loaders
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

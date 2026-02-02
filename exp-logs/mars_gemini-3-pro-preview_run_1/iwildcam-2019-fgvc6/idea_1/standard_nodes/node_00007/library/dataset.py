import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class AnimalDataset(Dataset):
    """
    Custom Dataset for Animal Classification.
    Reads images from disk based on metadata paths.
    """

    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        # Check if 'Category' column exists (it won't for test set usually)
        self.has_label = "Category" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., 'train_images/xxx.jpg')
        file_path = row["file_path"]
        full_path = os.path.join(self.root_dir, file_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label
        if self.has_label:
            label = row["Category"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Return dummy label for test set
            return image, torch.tensor(0, dtype=torch.long)


def get_transforms(mode="train"):
    """
    Returns the albumentations transform pipeline for train or val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3,
                ),
                A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                ToTensorV2(),
            ]
        )


def get_balanced_loader(df, root_dir, transform, batch_size, num_workers):
    """
    Creates a DataLoader with a WeightedRandomSampler to handle class imbalance.
    """
    # Calculate class counts
    class_counts = df["Category"].value_counts().sort_index()

    # Compute weight for each class (inverse frequency)
    # We map these weights to each sample in the dataframe
    class_weights = 1.0 / class_counts
    sample_weights = df["Category"].map(class_weights).values

    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights).double()

    # Create WeightedRandomSampler
    # num_samples=len(df) ensures the epoch size remains the same
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(df), replacement=True
    )

    dataset = AnimalDataset(df, root_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def get_loaders(debug=False, batch_size=Config.BATCH_SIZE):
    """
    Main function to initialize and return Train, Val, and Test loaders.
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    if debug:
        # Sample a small subset for debugging purposes
        df_train = df_train.sample(
            n=min(len(df_train), 500), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), 100), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), 100), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # 1. Train Loader (Standard with Shuffle)
    train_dataset = AnimalDataset(df_train, Config.INPUT_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Validation Loader (Standard)
    val_dataset = AnimalDataset(df_val, Config.INPUT_DIR, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test Loader (Standard)
    test_dataset = AnimalDataset(df_test, Config.INPUT_DIR, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A

from library.utils import seed_everything


class StegoDataset(Dataset):
    """
    Dataset class for Steganography Detection.
    Loads images, extracts the Y (Luminance) channel, applies augmentations,
    and normalizes to [0, 1].
    """

    def __init__(self, df, input_dir, transform=None):
        self.df = df
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path
        img_path = os.path.join(self.input_dir, row["image_path"])

        # Load image (BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to YCrCb and extract Y channel (Channel 0)
        # The embedding is primarily in the luminance channel.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image[:, :, 0]  # Shape: (H, W), dtype: uint8

        # Apply augmentations if provided
        if self.transform:
            augmented = self.transform(image=y_channel)
            y_channel = augmented["image"]

        # Preprocessing:
        # 1. Convert to float32
        # 2. Normalize to [0, 1]
        # 3. Add channel dimension -> (1, H, W)
        y_channel = y_channel.astype(np.float32) / 255.0
        y_channel = torch.from_numpy(y_channel).unsqueeze(0)

        # Get label
        label = torch.tensor(row["label"], dtype=torch.float32)

        return y_channel, label


def get_transforms(split="train"):
    """
    Returns the augmentation pipeline.
    Implements Dihedral group augmentations (D4) for training.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    else:
        return None


def get_dataloaders(
    input_dir="./input",
    metadata_dir="./metadata",
    batch_size=32,
    num_workers=4,
    seed=42,
):
    """
    Creates and returns Training and Validation DataLoaders.
    Implements WeightedRandomSampler for class balancing in the training set.
    """
    seed_everything(seed)

    # Load metadata
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            "Metadata files not found. Ensure ./metadata/train.csv and ./metadata/val.csv exist."
        )

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # --- Class Balancing Strategy ---
    # The dataset has 1 Cover image for every 3 Stego images (JMiPOD, JUNIWARD, UERD).
    # To optimize for the metric and training stability, we balance the batches 50/50.

    targets = train_df["label"].values
    class_counts = np.bincount(targets.astype(int))

    # Calculate weights: inverse of frequency
    # We guard against division by zero though counts should be > 0
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # Assign a weight to each sample corresponding to its class
    sample_weights = class_weights[targets.astype(int)]

    # Create WeightedRandomSampler
    # replacement=True allows oversampling of the minority class (Cover)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # --- Dataset Instantiation ---
    train_dataset = StegoDataset(
        df=train_df, input_dir=input_dir, transform=get_transforms("train")
    )

    val_dataset = StegoDataset(
        df=val_df, input_dir=input_dir, transform=get_transforms("val")
    )

    # --- DataLoader Instantiation ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
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

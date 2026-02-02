import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
DEFAULT_IMG_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RetinopathyDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy detection.
    Reads images via OpenCV, applies 'squashing' resize to 512x512,
    and converts labels to rank-consistent ordinal vectors.
    """

    def __init__(self, csv_path, transform=None, mode="train", sample_size=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            sample_size (int, optional): If provided, limits the dataset to this many samples for debugging.
        """
        self.mode = mode
        self.transform = transform

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Debugging: Subset if requested
        if sample_size is not None and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative, e.g., "train_images/id.png"
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Handle missing or corrupt images gracefully
        if image is None:
            # Create a black placeholder image of the expected size
            image = np.zeros((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            # For inference, we need the ID to build the submission
            return image, row["id_code"]
        else:
            # For train/val, we return image and ordinal label
            diagnosis = row["diagnosis"]

            # Convert integer label (0-4) to ordinal vector (size 4)
            # Label 0 -> [0, 0, 0, 0]
            # Label 1 -> [1, 0, 0, 0]
            # Label 2 -> [1, 1, 0, 0]
            # Label 3 -> [1, 1, 1, 0]
            # Label 4 -> [1, 1, 1, 1]
            target = np.zeros(4, dtype=np.float32)
            if diagnosis > 0:
                target[: int(diagnosis)] = 1.0

            return image, torch.tensor(target, dtype=torch.float32)


def get_transforms(img_size=DEFAULT_IMG_SIZE, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        img_size (int): Target spatial dimension (square).
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Simple "squashing" resize ignoring aspect ratio
                A.Resize(height=img_size, width=img_size),
                # Geometric Invariance Suite
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


def create_dataloaders(
    train_csv="./metadata/train.csv",
    val_csv="./metadata/val.csv",
    test_csv="./metadata/test.csv",
    batch_size=32,
    num_workers=4,
    img_size=DEFAULT_IMG_SIZE,
    sample_size=None,
    seed=42,
):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        train_csv (str): Path to train metadata.
        val_csv (str): Path to validation metadata.
        test_csv (str): Path to test metadata.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        img_size (int): Image resolution.
        sample_size (int, optional): Limit dataset size for debugging.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(seed)

    # Define Transforms
    train_transforms = get_transforms(img_size, mode="train")
    val_transforms = get_transforms(img_size, mode="val")

    # Initialize Datasets
    train_dataset = RetinopathyDataset(
        csv_path=train_csv,
        transform=train_transforms,
        mode="train",
        sample_size=sample_size,
    )

    val_dataset = RetinopathyDataset(
        csv_path=val_csv, transform=val_transforms, mode="val", sample_size=sample_size
    )

    test_dataset = RetinopathyDataset(
        csv_path=test_csv,
        transform=val_transforms,
        mode="test",
        # Usually we want full test set even in debug mode to ensure submission format is correct,
        # but if sample_size is very small for quick pipeline check, it applies here too.
        sample_size=sample_size,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

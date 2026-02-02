import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
IMG_SIZE = 260
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(split="train", img_size=IMG_SIZE):
    """
    Returns the Albumentations transform pipeline for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                # Strong augmentation strategy for high cardinality dataset
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    elif split == "val" or split == "test":
        return A.Compose(
            [
                # Deterministic transforms for evaluation
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown split: {split}")


class PlantDataset(Dataset):
    """
    Custom Dataset for Plant Classification.
    Reads images from disk and applies transforms.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            root_dir (str): Root directory for image files.
            transform (A.Compose): Albumentations transforms.
            is_test (bool): Whether this is a test set (returns image_id instead of label).
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        full_path = os.path.join(self.root_dir, file_path)

        # Read image using OpenCV
        image = cv2.imread(full_path)

        # Handle missing or corrupt images gracefully by returning a black image
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            fallback = A.Compose(
                [
                    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                    A.Normalize(mean=MEAN, std=STD),
                    ToTensorV2(),
                ]
            )
            image = fallback(image=image)["image"]

        if self.is_test:
            # Return image and image_id for submission generation
            return image, str(row["image_id"])
        else:
            # Return image and label
            label = row["category_id"]
            return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    train_batch_size=32,
    val_batch_size=64,
    num_workers=4,
    debug=False,
    img_size=IMG_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/testing.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, uses a small subset of data for debugging.
        img_size (int): Image resolution.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    set_seed(42)

    # Load metadata
    train_csv = os.path.join("./metadata", "train.csv")
    val_csv = os.path.join("./metadata", "val.csv")
    test_csv = os.path.join("./metadata", "test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Debug mode: sample a small subset to speed up development
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), 2000), random_state=42
        ).reset_index(drop=True)
        val_df = val_df.sample(n=min(len(val_df), 500), random_state=42).reset_index(
            drop=True
        )
        test_df = test_df.sample(n=min(len(test_df), 500), random_state=42).reset_index(
            drop=True
        )

    # Define transforms
    train_transform = get_transforms("train", img_size)
    val_transform = get_transforms("val", img_size)

    # Instantiate Datasets
    train_dataset = PlantDataset(
        train_df, INPUT_DIR, transform=train_transform, is_test=False
    )
    val_dataset = PlantDataset(
        val_df, INPUT_DIR, transform=val_transform, is_test=False
    )
    test_dataset = PlantDataset(
        test_df, INPUT_DIR, transform=val_transform, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

import os
import cv2
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config


class DogCatDataset(Dataset):
    """
    Custom Dataset for Dog vs Cat classification.
    Reads images from disk and applies transformations.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Albumentations transform pipeline.
            is_test (bool): Whether this is the test set (returns ID instead of label).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepath is relative to INPUT_DIR (e.g., "train/cat.0.jpg")
        img_path = os.path.join(self.root_dir, row["filepath"])

        # Read image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission
            return image, row["id"]
        else:
            # Return image and label (float for BCEWithLogitsLoss)
            return image, torch.tensor(row["label"], dtype=torch.float32)


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'val'/'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    scale=(Config.CROP_SCALE_MIN, Config.CROP_SCALE_MAX),
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def create_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Define transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Instantiate Datasets
    train_dataset = DogCatDataset(
        train_df, Config.INPUT_DIR, transform=train_transform, is_test=False
    )
    val_dataset = DogCatDataset(
        val_df, Config.INPUT_DIR, transform=val_transform, is_test=False
    )
    test_dataset = DogCatDataset(
        test_df, Config.INPUT_DIR, transform=val_transform, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

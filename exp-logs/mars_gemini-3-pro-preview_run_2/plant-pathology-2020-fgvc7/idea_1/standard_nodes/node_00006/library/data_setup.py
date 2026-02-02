import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything


class AppleLeafDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Loads images from disk and processes them with albumentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata DataFrame.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'. Determines output format.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-calculate full file paths
        # Metadata file_path is relative (e.g., 'images/Train_0.jpg')
        # Config.INPUT_ROOT is './input'
        self.file_paths = (
            self.df["file_path"]
            .apply(lambda x: os.path.join(Config.INPUT_ROOT, x))
            .values
        )

        self.image_ids = self.df["image_id"].values

        # Extract labels for train/val modes
        if self.mode != "test":
            self.labels = self.df[Config.TARGET_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            return image, self.image_ids[idx]
        else:
            # Convert one-hot encoding to class index for CrossEntropyLoss
            # argmax returns the index of the max value (the class 1)
            label_idx = np.argmax(self.labels[idx])
            label = torch.tensor(label_idx, dtype=torch.long)
            return image, label


def get_transforms(data="train"):
    """
    Returns albumentations transforms for specific data modes.

    Args:
        data (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=72,
                    max_width=72,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data in ["val", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def create_dataloaders(debug=False):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, subsets the data for faster iteration.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Apply Debug Subsetting
    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Initialize Datasets
    train_dataset = AppleLeafDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AppleLeafDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_dataset = AppleLeafDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # Initialize DataLoaders
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

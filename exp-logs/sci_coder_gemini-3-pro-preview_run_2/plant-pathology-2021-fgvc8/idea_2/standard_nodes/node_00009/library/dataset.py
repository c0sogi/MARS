import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(data: str):
    """
    Returns the Albumentations transformation pipeline based on the data split.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data type: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles loading images via OpenCV and converting labels to binary vectors.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'image', 'labels', 'file_path'.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.classes = Config.CLASSES
        # Create a mapping from class name to index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path. Metadata file_path is relative to input dir.
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, return image and image_id (for submission)
            return image, row["image"]
        else:
            # For train/valid, return image and multi-hot encoded target
            label_str = row["labels"]
            # Labels are space-delimited
            labels = label_str.split()

            # Initialize zero vector
            target = torch.zeros(len(self.classes), dtype=torch.float32)

            for label in labels:
                if label in self.class_to_idx:
                    target[self.class_to_idx[label]] = 1.0

            return image, target


def get_loaders():
    """
    Initializes and returns DataLoaders for train, validation, and test sets.
    Reads metadata from paths defined in Config.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Sample a subset of data
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} images per split.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Initialize Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"), mode="valid")
    test_dataset = AppleDataset(test_df, transforms=get_transforms("test"), mode="test")

    # Initialize DataLoaders
    # Drop last for training to maintain consistent batch statistics
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

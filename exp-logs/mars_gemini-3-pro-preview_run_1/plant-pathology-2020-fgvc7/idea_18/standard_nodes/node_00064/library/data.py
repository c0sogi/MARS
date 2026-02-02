import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Valid and Test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.transforms = transforms
        self.target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

        # Check if targets exist in dataframe (they won't for test set)
        self.has_labels = all(col in df.columns for col in self.target_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "images/Train_0.jpg"
        # Config.INPUT_DIR is "./input"
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        # Handle Labels
        if self.has_labels:
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # Return dummy labels for test set to maintain signature consistency
            return image, torch.zeros(len(self.target_cols), dtype=torch.float32)


def get_loaders():
    """
    Creates DataLoaders for training and validation using fixed metadata files.
    Cite solution_lesson_node_00058: Preference for Seed Averaging over Bagging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        train_df = train_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(train_df, transforms=get_transforms("train"))
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"))

    # Create Loaders
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
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    df = pd.read_csv(Config.TEST_METADATA_PATH)

    dataset = AppleDataset(df, transforms=get_transforms("test"))

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader

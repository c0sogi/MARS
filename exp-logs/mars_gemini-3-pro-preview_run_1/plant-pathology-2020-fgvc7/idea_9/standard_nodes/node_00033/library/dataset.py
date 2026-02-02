import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles loading images and labels based on the provided metadata DataFrame.
    """

    def __init__(self, df, transforms=None, mode="train", root_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, labels).
            transforms (albumentations.Compose): Transformations to apply to the images.
            mode (str): 'train' (returns image, label) or 'test' (returns image, image_id).
            root_dir (str): Root directory where images are located.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.root_dir = root_dir
        self.class_labels = Config.CLASS_LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "images/Train_0.jpg")
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode == "test":
            # In test mode, return image and image_id for submission mapping
            return image, row["image_id"]
        else:
            # In train mode, return image and class index label
            # Extract probability/one-hot columns
            labels = row[self.class_labels].values.astype(np.float32)
            # Convert to class index for CrossEntropyLoss (0 to Num_Classes-1)
            label_idx = np.argmax(labels)

            return image, torch.tensor(label_idx, dtype=torch.long)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train' for augmentation, 'valid'/'test' for resizing/normalization only.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                # VerticalFlip is critical for leaf images (rotationally invariant)
                A.VerticalFlip(p=0.5),
                # Increased rotation limit to 30 degrees (Cite Lesson 00030)
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms
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


def get_datasets(use_full_data=True):
    """
    Factory function to create train, validation, and test datasets.

    Args:
        use_full_data (bool): If True, merges train and validation metadata for training
                              (Strategy: Full-Dataset Training). Returns None for val_dataset.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Implement Full-Dataset Strategy
    if use_full_data:
        # Concatenate train and validation sets to use 100% of data
        train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        val_df = None  # No validation set available in this mode

    # Debug Subsampling
    if Config.DEBUG:
        train_df = train_df.sample(
            frac=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        if val_df is not None:
            val_df = val_df.sample(
                frac=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)
        test_df = test_df.sample(
            frac=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = AppleDataset(
        df=train_df, transforms=get_transforms(mode="train"), mode="train"
    )

    val_dataset = None
    if val_df is not None:
        val_dataset = AppleDataset(
            df=val_df,
            transforms=get_transforms(mode="valid"),
            mode="train",  # Returns labels
        )

    test_dataset = AppleDataset(
        df=test_df,
        transforms=get_transforms(mode="test"),
        mode="test",  # Returns image_id
    )

    return train_dataset, val_dataset, test_dataset

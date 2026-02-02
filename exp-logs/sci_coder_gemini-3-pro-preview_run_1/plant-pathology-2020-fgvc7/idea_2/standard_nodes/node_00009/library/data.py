import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles loading images, applying augmentations, and extracting labels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image_id and optionally target columns.
            transforms (albumentations.Compose): Albumentations transforms to apply.
            mode (str): 'train' or 'test'. If 'train', expects target columns.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.labels = Config.CLASS_LABELS

        # Pre-check if label columns exist for training mode
        if self.mode == "train":
            missing_cols = [col for col in self.labels if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing target columns in dataframe: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input directory (e.g., "images/Train_0.jpg")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to basic tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        # Handle Labels
        if self.mode == "train":
            # Extract one-hot/probability vector and convert to class index
            # Assumes single-label classification for CrossEntropyLoss
            label_probs = row[self.labels].values.astype(float)
            label_idx = np.argmax(label_probs)
            return image, torch.tensor(label_idx, dtype=torch.long)
        else:
            # For test mode, return image and image_id
            return image, row["image_id"]


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' or 'valid'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
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


def load_full_train_data():
    """
    Loads and merges the training and validation metadata to form the full training set
    for Cross-Validation.

    Returns:
        pd.DataFrame: Combined dataframe.
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine them
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    return full_df


def load_test_data():
    """
    Loads the test metadata.

    Returns:
        pd.DataFrame: Test dataframe.
    """
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    return pd.read_csv(Config.TEST_METADATA_PATH)

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Mild augmentation as per baseline strategy to handle scanner variations
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter Detection.
    Handles loading images, replicating channels to RGB, and extracting multi-label targets.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (paths and labels).
            transforms (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Used for debug logic.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Handle Debugging: Sample a subset if Config.DEBUG is True
        if Config.DEBUG:
            # Deterministic sampling for reproducibility
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        self.file_paths = self.df["file_path"].values

        # Pre-fetch labels. Config.TARGET_COLS ensures we get the correct 11 columns in order.
        # Note: Test metadata also contains these columns (initialized to 0).
        if all(col in self.df.columns for col in Config.TARGET_COLS):
            self.labels = self.df[Config.TARGET_COLS].values.astype(np.float32)
        else:
            # Fallback should not be hit given the metadata structure, but ensures safety
            self.labels = np.zeros((len(self.df), Config.NUM_CLASSES), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Construct File Path
        # Metadata file_path is relative to input dir (e.g., "train/UID.jpg")
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # 2. Load Image
        # Load as BGR (default) to ensure 8-bit depth and 3 channels.
        # If the source X-ray is grayscale, OpenCV replicates it to 3 channels automatically.
        img = cv2.imread(img_path)

        if img is None:
            # Raise error if image is unreadable
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")

        # 3. Convert BGR to RGB
        # Albumentations and Pre-trained models expect RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # 5. Get Labels
        label = self.labels[idx]

        return img, torch.tensor(label, dtype=torch.float32)

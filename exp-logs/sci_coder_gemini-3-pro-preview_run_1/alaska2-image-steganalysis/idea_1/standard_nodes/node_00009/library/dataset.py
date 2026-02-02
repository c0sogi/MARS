import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class StegoDataset(Dataset):
    """
    PyTorch Dataset for Steganography Detection.
    Reads images from disk, applies rigid augmentations, and returns tensors.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
            transform (albumentations.Compose): Albumentations transform pipeline.
        """
        self.df = df
        self.transform = transform
        self.root = Config.input_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path is relative, e.g., "Cover/00001.jpg"
        img_path = os.path.join(self.root, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label (default to 0 if not present, e.g., for test set inference structure)
        label = row["label"] if "label" in row else 0

        # Return image tensor and label as float tensor for BCEWithLogitsLoss
        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(mode="train", img_size=None):
    """
    Generates the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): Target image size (height and width).
                        Defaults to Config.img_size if None.

    Returns:
        A.Compose: The transform pipeline.
    """
    if img_size is None:
        img_size = Config.img_size

    if mode == "train":
        return A.Compose(
            [
                # RandomCrop ensures fixed input size.
                # Since most images are 512x512, this is effectively a no-op or a safety check.
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                # Rigid augmentations that preserve the pixel grid / DCT coefficients structure
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization for EfficientNet backbone (ImageNet stats)
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
                # CenterCrop for deterministic evaluation
                A.CenterCrop(height=img_size, width=img_size, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def load_metadata(csv_path, debug=False, subset_size=None):
    """
    Loads the metadata CSV file and optionally subsets it for debugging.

    Args:
        csv_path (str): Path to the CSV file.
        debug (bool): Whether to run in debug mode (subsetting data).
        subset_size (int): Number of samples to load in debug mode.

    Returns:
        pd.DataFrame: The loaded (and potentially subsetted) DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if debug and subset_size is not None:
        # Ensure we don't try to sample more than available
        n_samples = min(len(df), subset_size)

        # Shuffle and sample to get a random subset
        df = df.sample(n=n_samples, random_state=Config.seed).reset_index(drop=True)

        # Print info about the subset
        print(
            f"DEBUG MODE: Loaded subset of {len(df)} samples from {os.path.basename(csv_path)}"
        )

    return df

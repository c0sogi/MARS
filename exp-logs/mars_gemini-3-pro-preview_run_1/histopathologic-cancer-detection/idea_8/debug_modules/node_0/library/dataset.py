import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(split: str) -> A.Compose:
    """
    Returns the Albumentations transform pipeline for a given data split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose(
            [
                # Hard Attention: Crop center 48x48 from 96x96 input
                A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Color Augmentations (Mild, preserving stain colors)
                # Explicitly excluding HueSaturationValue
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test pipeline
        return A.Compose(
            [
                # Hard Attention: Crop center 48x48
                A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology Tumor Detection.
    """

    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'id', 'file_path', and optionally 'label'.
            transforms (A.Compose, optional): Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Construct full file path
        # file_path in metadata is relative to input dir (e.g., "train/xxxx.tif")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get label if it exists (Train/Val), else dummy (Test)
        if "label" in row:
            label = torch.tensor(row["label"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)

        # Return image, label, and id (useful for tracking/submission)
        return image, label, row["id"]


def load_dataset(split: str, debug: bool = Config.DEBUG) -> PathologyDataset:
    """
    Factory function to load metadata and create a PathologyDataset.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        debug (bool): If True, subsamples the dataset for debugging.

    Returns:
        PathologyDataset: The instantiated dataset.
    """
    # Select metadata path based on split
    if split == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        csv_path = Config.VAL_METADATA_PATH
    elif split == "test":
        csv_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    # Load metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle Debugging / Subsampling
    if debug:
        max_samples = (
            Config.MAX_TEST_SAMPLES if split == "test" else Config.MAX_TRAIN_SAMPLES
        )
        if max_samples is not None and len(df) > max_samples:
            # Stratified sampling for train/val if label exists
            if "label" in df.columns:
                df = df.groupby("label", group_keys=False).apply(
                    lambda x: x.sample(
                        n=min(len(x), int(max_samples / 2)), random_state=Config.SEED
                    )
                )
                # Shuffle the result
                df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
            else:
                # Simple random sampling for test
                df = df.sample(n=max_samples, random_state=Config.SEED).reset_index(
                    drop=True
                )

            print(f"DEBUG MODE: Subsampled {split} dataset to {len(df)} samples.")

    # Create Dataset
    transforms = get_transforms(split)
    dataset = PathologyDataset(df, transforms=transforms)

    return dataset

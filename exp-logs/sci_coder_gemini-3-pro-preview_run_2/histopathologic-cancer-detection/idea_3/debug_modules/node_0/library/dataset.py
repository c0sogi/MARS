import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from library.config import Config


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for loading pathology image patches.
    Reads metadata from CSV files and loads images on-the-fly.
    """

    def __init__(
        self,
        metadata_path: str,
        transform: Optional[A.Compose] = None,
        sample_size: Optional[int] = None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (A.Compose, optional): Albumentations transform pipeline.
            sample_size (int, optional): Limit dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.transform = transform

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Debugging: Sample subset if requested
        if sample_size is not None and sample_size < len(self.df):
            # Deterministic sampling for reproducibility
            self.df = self.df.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

        # Pre-fetch paths and labels to avoid dataframe overhead in __getitem__
        self.file_paths = self.df["file_path"].values
        self.labels = self.df["label"].values
        self.ids = self.df["id"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image using OpenCV
        # cv2.imread returns BGR
        image = cv2.imread(full_path)

        if image is None:
            # Handle missing/corrupt images gracefully by returning a blank image
            # This ensures the dataloader doesn't crash during training
            image = np.zeros(
                (Config.FULL_IMAGE_SIZE, Config.FULL_IMAGE_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label (float32 for BCEWithLogitsLoss)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def get_transforms(mode: str = "train") -> A.Compose:
    """
    Constructs the Albumentations transform pipeline.

    Strategy:
    1. Global Augmentations (Train only) on full 96x96 patch.
    2. Center Crop to 64x64 (All modes).
    3. Normalize and Convert to Tensor (All modes).

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms = []

    # 1. Global Augmentations (Train only)
    # Applied to the full 96x96 context before cropping
    if mode == "train":
        transforms.extend(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Intensity Augmentations (ColorJitter)
                # Helps with stain variability and lighting differences
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
            ]
        )

    # 2. Contextual Crop (All modes)
    # Crop 64x64 from the center of the 96x96 image
    transforms.append(A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE))

    # 3. Normalization & Tensor Conversion
    transforms.extend(
        [
            A.Normalize(
                mean=Config.DATASET_MEAN,
                std=Config.DATASET_STD,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def get_loaders(
    debug: bool = Config.DEBUG,
    sample_size: int = Config.DEBUG_SAMPLE_SIZE,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test splits.

    Args:
        debug (bool): Whether to run in debug mode (subset of data).
        sample_size (int): Number of samples to use in debug mode.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """

    # Determine effective sample size
    limit = sample_size if debug else None

    # Create Datasets
    train_dataset = PathologyDataset(
        metadata_path=Config.TRAIN_METADATA,
        transform=get_transforms(mode="train"),
        sample_size=limit,
    )

    val_dataset = PathologyDataset(
        metadata_path=Config.VAL_METADATA,
        transform=get_transforms(mode="val"),
        sample_size=limit,
    )

    test_dataset = PathologyDataset(
        metadata_path=Config.TEST_METADATA,
        transform=get_transforms(mode="test"),
        sample_size=limit,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for training stability with Mixup/BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

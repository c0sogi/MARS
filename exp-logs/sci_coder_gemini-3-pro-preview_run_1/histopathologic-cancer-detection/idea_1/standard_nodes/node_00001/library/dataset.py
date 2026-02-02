import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    # Base transforms: Center Crop to 48x48 and Normalize to [0, 1]
    # Note: Normalize with mean=0, std=1, max_pixel_value=255.0 is equivalent to dividing by 255.0
    transforms_list = [
        A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
    ]

    if phase == "train":
        # Add geometric augmentations for training
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # Convert to PyTorch Tensor (HWC -> CHW)
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class PathologyDataset(Dataset):
    """
    Custom Dataset for loading Pathology images.
    """

    def __init__(self, df: pd.DataFrame, phase: str, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, label).
            phase (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.phase = phase
        self.transform = transform

        # Pre-calculate full paths to avoid doing it in __getitem__ repeatedly
        # file_path in metadata is relative to input dir (e.g., "train/xxx.tif")
        self.image_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        # Store labels if available
        if "label" in df.columns:
            self.labels = df["label"].values.astype(np.float32)
        else:
            self.labels = None

        # Store IDs for test set
        self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image using OpenCV
        # cv2.imread loads as BGR, we need RGB
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (should not happen based on metadata check)
            # Return a black image of expected size (96x96 original)
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (though get_transforms should be used)
            # Just convert to tensor and scale
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.phase == "test":
            return image, self.ids[idx]
        else:
            label = self.labels[idx]
            return image, label


def get_dataloaders(
    train_csv: str = Config.TRAIN_CSV,
    val_csv: str = Config.VAL_CSV,
    test_csv: str = Config.TEST_CSV,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    debug_sample_size: int = Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_csv (str): Path to train metadata CSV.
        val_csv (str): Path to validation metadata CSV.
        test_csv (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        debug_sample_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Debug Sampling
    if debug_sample_size is not None:
        if len(train_df) > debug_sample_size:
            train_df = train_df.sample(
                n=debug_sample_size, random_state=Config.SEED
            ).reset_index(drop=True)
        if len(val_df) > debug_sample_size:
            val_df = val_df.sample(
                n=debug_sample_size, random_state=Config.SEED
            ).reset_index(drop=True)
        if len(test_df) > debug_sample_size:
            test_df = test_df.sample(
                n=debug_sample_size, random_state=Config.SEED
            ).reset_index(drop=True)
        print(
            f"Debug Mode: Sampled {len(train_df)} train, {len(val_df)} val, {len(test_df)} test images."
        )

    # Create Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    # Create Datasets
    train_dataset = PathologyDataset(train_df, phase="train", transform=train_transform)
    val_dataset = PathologyDataset(val_df, phase="val", transform=val_transform)
    test_dataset = PathologyDataset(test_df, phase="test", transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

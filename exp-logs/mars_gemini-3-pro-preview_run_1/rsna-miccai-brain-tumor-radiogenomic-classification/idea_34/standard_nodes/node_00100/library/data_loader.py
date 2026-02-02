import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    MODALITIES,
    ROI_RELATIVE_DEPTHS,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import get_logger
from library.roi_processing import generate_roi_cache, read_dicom

# Initialize Logger
logger = get_logger("data_loader")


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for Glioblastoma MGMT prediction.
    Constructs a 9-channel volumetric input tensor based on relative ROI depths.
    """

    def __init__(self, df, transform=None, is_train=False):
        self.df = df
        self.transform = transform
        self.is_train = is_train

        # Pre-compute the list of column keys for the 9 channels
        # Order: Depth 0.4 [FLAIR, T1wCE, T2w], Depth 0.5 [...], Depth 0.6 [...]
        self.channel_keys = []
        for depth in ROI_RELATIVE_DEPTHS:
            for mod in MODALITIES:
                # Key format matches roi_processing output: e.g., "FLAIR_0.4_path"
                key = f"{mod}_{depth}_path"
                self.channel_keys.append(key)

    def __len__(self):
        return len(self.df)

    def load_slice(self, rel_path):
        """
        Reads a DICOM file, resizes it, and applies Min-Max normalization.
        """
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Use the robust read_dicom from library
        img = read_dicom(full_path)

        if img is None:
            # Fallback for missing data (should be caught by integrity checks, but safe to handle)
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Resize to target dimension
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Independent Min-Max Normalization to [0, 1]
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        channels = []

        # Load all 9 channels strictly in order
        for key in self.channel_keys:
            rel_path = row[key]
            slice_img = self.load_slice(rel_path)
            channels.append(slice_img)

        # Stack to create (H, W, C) -> (224, 224, 9)
        image = np.stack(channels, axis=-1)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return tuple based on whether target exists
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image, target
        else:
            # Test set (return BraTS21ID for tracking if needed, but standard is just image)
            return image


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transform pipeline.
    Implements Spatially-Preserved Augmentation (no translation/scaling).
    """
    if data_type == "train":
        return A.Compose(
            [
                # Spatially-Preserved Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Deformations that preserve centroid
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                # Convert to Tensor (C, H, W)
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Tensor conversion
        return A.Compose([ToTensorV2()])


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to load cached ROI paths or recompute.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # 1. Get processed dataframes (with ROI paths)
    df_train, df_val, df_test = generate_roi_cache(load_cached_data=load_cached_data)

    logger.info(f"Creating DataLoaders with Batch Size: {BATCH_SIZE}")

    # 2. Define Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # 3. Create Datasets
    train_dataset = BraTSDataset(df_train, transform=train_transform, is_train=True)
    val_dataset = BraTSDataset(df_val, transform=val_transform, is_train=False)
    test_dataset = BraTSDataset(df_test, transform=val_transform, is_train=False)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

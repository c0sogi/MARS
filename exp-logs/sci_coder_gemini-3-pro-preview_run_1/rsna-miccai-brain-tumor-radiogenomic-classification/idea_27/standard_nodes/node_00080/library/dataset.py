import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    MODALITIES,
    RELATIVE_DEPTHS,
    BATCH_SIZE,
    SEED,
)
from library.image_processing import load_volumetric_stack


def get_transforms(phase: str):
    """
    Returns the Albumentations composition for the specified phase.

    Strategy:
    - Train: Spatially-Preserved Augmentations (Elastic, Grid, Rotate, Flip).
             Strictly excludes Translation and Scaling to preserve Centroid alignment.
    - Valid/Test: Only tensor conversion.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate without shifting or scaling to maintain ROI alignment
                A.Rotate(limit=15, p=0.5, border_mode=0),
                # Elastic and Grid distortions for volumetric texture robustness
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.3, border_mode=0
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3, border_mode=0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 MGMT Promoter Methylation Prediction.
    Implements the Scale-Invariant Relative-Volumetric (SIRV) sampling strategy.
    """

    def __init__(self, df: pd.DataFrame, phase: str = "train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'BraTS21ID', 'subject_path', etc.
            phase (str): 'train', 'valid', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform

        # Configuration for loading logic
        self.modalities = MODALITIES
        self.relative_depths = RELATIVE_DEPTHS
        self.img_size = IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Construct full path to subject directory
        # metadata contains relative path e.g., "train/00000"
        subject_dir = os.path.join(INPUT_DIR, row["subject_path"])

        # Load the 9-channel volumetric tensor (H, W, 9)
        # This handles ROI detection, relative depth sampling, and caching internally.
        image = load_volumetric_stack(
            subject_id=subject_id,
            subject_dir=subject_dir,
            modalities=self.modalities,
            relative_depths=self.relative_depths,
            img_size=self.img_size,
            load_cached_data=True,
        )

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen with get_transforms)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return data based on phase
        if self.phase == "test":
            # For test, we need the ID to map predictions
            return image, str(subject_id)
        else:
            # For train/val, we return image and label
            label = row["MGMT_value"]
            return image, torch.tensor(label, dtype=torch.float)


def get_dataloader(
    df: pd.DataFrame,
    phase: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    shuffle: bool = None,
):
    """
    Factory function to create a DataLoader for a specific phase.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        phase (str): 'train', 'valid', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        shuffle (bool): Whether to shuffle data. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    transform = get_transforms(phase)
    dataset = BraTSDataset(df, phase=phase, transform=transform)

    if shuffle is None:
        shuffle = phase == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(phase == "train"),  # Drop last incomplete batch only during training
    )

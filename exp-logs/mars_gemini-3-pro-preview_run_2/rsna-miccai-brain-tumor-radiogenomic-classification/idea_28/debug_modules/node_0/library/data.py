import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import (
    read_dicom_robust,
    resize_image,
    normalize_min_max,
    get_sorted_image_files,
    generate_roi_cache,
)


class MGMTDataset(Dataset):
    """
    Dataset class for GLioblastoma MGMT promoter methylation prediction.
    Implements Hierarchical-Gated ROI Selection, Stacking, and Augmentation.
    """

    def __init__(self, metadata_df, transform=None, load_cached_roi=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing subject metadata.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_roi (bool): Whether to use cached ROI anchor indices.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

        # Generate or load the ROI anchor cache
        # This returns a dict: {str(BraTS21ID): int(anchor_index)}
        self.roi_cache = generate_roi_cache(
            self.metadata_df, load_cached_data=load_cached_roi
        )

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        subject_id = str(row["BraTS21ID"])

        # Get label (if available, else 0.0 for test)
        label = row["MGMT_value"] if "MGMT_value" in row else 0.0

        # Retrieve Anchor Index
        anchor_idx = self.roi_cache.get(subject_id, 0)

        # Define Slice Indices (Anchor +/- Stride)
        # We want 3 slices: [Anchor - Stride, Anchor, Anchor + Stride]
        stride = Config.STRIDE
        relative_indices = [-stride, 0, stride]

        # Initialize list to hold all 12 channels (4 modalities * 3 slices)
        # Order must match Config.MODALITIES: FLAIR, T1w, T1wCE, T2w
        # This aligns with the Grouped Conv Stem in the model
        channels = []

        for mod in Config.MODALITIES:
            path_col = f"path_{mod}"
            dir_path = os.path.join(Config.INPUT_DIR, row[path_col])

            # Get sorted files for this modality
            files = get_sorted_image_files(dir_path)
            num_files = len(files)

            # Process the 3 required slices
            for rel_idx in relative_indices:
                target_idx = anchor_idx + rel_idx

                # Edge Clamping
                if num_files > 0:
                    if target_idx < 0:
                        target_idx = 0
                    elif target_idx >= num_files:
                        target_idx = num_files - 1

                    # Read Image
                    file_path = os.path.join(dir_path, files[target_idx])
                    img = read_dicom_robust(file_path)
                else:
                    # Fallback for missing modality/empty folder
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

                # Preprocessing
                img = resize_image(img, size=Config.IMG_SIZE)
                img = normalize_min_max(img)

                channels.append(img)

        # Stack channels to create volume: (H, W, C=12)
        volume = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented["image"]

        # Convert to Tensor (H, W, C) -> (C, H, W)
        # If ToTensorV2 is used in transform, it handles this.
        # If not (e.g. custom pipeline), we do it manually.
        # Here we assume transform returns numpy or tensor.
        # Standard Albumentations returns numpy unless ToTensorV2 is last.

        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).permute(2, 0, 1).float()
        elif isinstance(volume, torch.Tensor):
            # Ensure float32
            volume = volume.float()

        return volume, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the augmentation pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.Rotate(
                    limit=Config.ROTATION_DEGREES,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # No ToTensorV2 here to keep it flexible, handled in __getitem__
            ]
        )
    else:
        # No augmentation for validation/test
        return A.Compose([])


def get_dataloader(
    metadata_df,
    phase="train",
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create DataLoaders.
    """
    transform = get_transforms(phase)

    # For validation/test, we usually want deterministic behavior, so we can rely on cache
    # For training, we also want cache to speed up initialization
    dataset = MGMTDataset(metadata_df, transform=transform, load_cached_roi=True)

    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=shuffle,  # Drop last incomplete batch only during training
    )

    return loader

import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.data_processing import read_dicom_robust, resize_image, normalize_image
from library.roi_selection import generate_roi_cache
from library.utils import get_logger

logger = get_logger("dataset")


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                # Rotation with Reflection Padding to avoid artificial edges
                A.Rotate(
                    limit=Config.AUG_ROTATION, border_mode=cv2.BORDER_REFLECT, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Glioblastoma Subtype Prediction.
    Loads 12-channel MRI volumes based on Logical-Consensus ROI selection.
    """

    def __init__(self, metadata_df, roi_cache_df, transform=None, is_test=False):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing BraTS21ID and targets.
            roi_cache_df (pd.DataFrame): Cache containing file paths for each channel.
            transform (A.Compose, optional): Albumentations transforms.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.roi_cache = roi_cache_df
        self.transform = transform
        self.is_test = is_test

        # Create a quick lookup for file paths
        # Index by BraTS21ID for O(1) access
        if "BraTS21ID" in self.roi_cache.columns:
            self.path_lookup = self.roi_cache.set_index("BraTS21ID")
        else:
            self.path_lookup = self.roi_cache

        # Filter metadata to ensure we have cache entries (robustness)
        valid_ids = set(self.path_lookup.index)
        original_len = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata["BraTS21ID"].isin(valid_ids)
        ].reset_index(drop=True)

        if len(self.metadata) < original_len:
            logger.warning(
                f"Dropped {original_len - len(self.metadata)} subjects due to missing ROI cache entries."
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        subject_id = int(row["BraTS21ID"])

        # Retrieve file paths from cache
        try:
            paths_row = self.path_lookup.loc[subject_id]
        except KeyError:
            # Should not happen due to init filtering, but safe fallback
            logger.error(f"Subject {subject_id} not found in ROI cache.")
            volume = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], Config.INPUT_CHANNELS),
                dtype=np.float32,
            )
            if self.transform:
                volume = self.transform(image=volume)["image"]
            else:
                volume = torch.from_numpy(volume).permute(2, 0, 1)

            if self.is_test:
                return volume, subject_id
            else:
                return volume, torch.tensor(0.0, dtype=torch.float32)

        # Load and stack channels
        channels = []
        # Order: FLAIR (0,1,2), T1w (0,1,2), T1wCE (0,1,2), T2w (0,1,2)
        # This aligns with the Stem Groups=4 logic
        for mod in Config.INPUT_MODALITIES:
            for i in range(Config.NUM_SLICES_PER_MODALITY):
                col_name = f"{mod}_{i}"
                file_path = paths_row[col_name]

                # Load, Resize, Normalize
                # read_dicom_robust handles missing files by returning zeros
                img = read_dicom_robust(file_path)
                img = resize_image(img, Config.IMG_SIZE)
                img = normalize_image(img)

                channels.append(img)

        # Stack to (H, W, C)
        volume = np.stack(channels, axis=-1)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented["image"]  # Returns Tensor (C, H, W) due to ToTensorV2
        else:
            # Manual conversion if no transform provided
            volume = torch.from_numpy(volume).permute(2, 0, 1)

        if self.is_test:
            return volume, subject_id
        else:
            target = row["MGMT_value"]
            return volume, torch.tensor(target, dtype=torch.float32)


def create_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA,
    val_metadata_path=Config.VAL_METADATA,
    test_metadata_path=Config.TEST_METADATA,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Factory function to create DataLoaders for train, val, and test.
    Handles global cache generation to ensure consistency.
    """
    # 1. Load Metadata
    train_df = (
        pd.read_csv(train_metadata_path)
        if os.path.exists(train_metadata_path)
        else pd.DataFrame()
    )
    val_df = (
        pd.read_csv(val_metadata_path)
        if os.path.exists(val_metadata_path)
        else pd.DataFrame()
    )
    test_df = (
        pd.read_csv(test_metadata_path)
        if os.path.exists(test_metadata_path)
        else pd.DataFrame()
    )

    # 2. Generate/Load ROI Cache
    # Combine all IDs to ensure cache covers everything
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if combined_df.empty:
        raise ValueError("No metadata found. Check metadata paths.")

    roi_cache = generate_roi_cache(combined_df, load_cached_data=load_cached_data)

    dataloaders = {}

    # 3. Create Train Loader
    if not train_df.empty:
        train_ds = BrainTumorDataset(
            train_df, roi_cache, transform=get_transforms("train"), is_test=False
        )
        dataloaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
        )

    # 4. Create Val Loader
    if not val_df.empty:
        val_ds = BrainTumorDataset(
            val_df, roi_cache, transform=get_transforms("valid"), is_test=False
        )
        dataloaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # 5. Create Test Loader
    if not test_df.empty:
        test_ds = BrainTumorDataset(
            test_df, roi_cache, transform=get_transforms("test"), is_test=True
        )
        dataloaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders

import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from typing import Tuple, Dict

from library.config import Config
from library.data_processing import load_patient_volume, get_roi_cache


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor Classification using MRI scans.

    Implements:
    - Integration with Integral-ROI pipeline via anchor cache.
    - Modality-Isolated volume loading.
    - Geometric augmentations for training (Flip, Rotate).
    """

    def __init__(
        self, df: pd.DataFrame, roi_cache: Dict[str, int], phase: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing subject metadata and paths.
            roi_cache (Dict[str, int]): Dictionary mapping BraTS21ID (str) to anchor slice index.
            phase (str): One of 'train', 'val', 'test'. Controls augmentation and label return.
        """
        self.df = df
        self.roi_cache = roi_cache
        self.phase = phase

        # Define Augmentations for Training
        # Requirements: HorizontalFlip, VerticalFlip, RandomRotation +/- 15 degrees
        if self.phase == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Rotate limit is in degrees. value=0 fills border with black.
                    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                ]
            )
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]

        # Ensure ID is string for consistent cache lookup
        subject_id_str = str(row["BraTS21ID"])

        # 1. Get Anchor Index
        # Default to 0 if not in cache (should not happen if cache is built correctly)
        anchor_idx = self.roi_cache.get(subject_id_str, 0)

        # 2. Load Volume
        # Returns Tensor of shape (C, H, W) -> (12, 224, 224)
        tensor = load_patient_volume(row, anchor_idx)

        # 3. Apply Augmentations
        if self.transform:
            # Albumentations works with HWC numpy arrays
            data_np = tensor.numpy()  # (C, H, W)
            data_np = np.transpose(data_np, (1, 2, 0))  # (H, W, C)

            # Apply transform
            augmented = self.transform(image=data_np)["image"]

            # Convert back to CHW Tensor
            data_np = np.transpose(augmented, (2, 0, 1))
            tensor = torch.from_numpy(data_np)

        # 4. Get Label
        if "MGMT_value" in row:
            label_val = row["MGMT_value"]
            label = torch.tensor(label_val, dtype=torch.float32)
        else:
            # Test set does not have labels; return dummy
            label = torch.tensor(-1.0, dtype=torch.float32)

        return tensor, label, subject_id_str


def get_datasets(
    load_cached_data: bool = True,
) -> Tuple[BrainTumorDataset, BrainTumorDataset, BrainTumorDataset]:
    """
    Factory function to initialize datasets for all splits.

    Handles the loading of metadata and the generation/loading of the ROI cache.

    Args:
        load_cached_data (bool): If True, attempts to load ROI cache from disk.
                                 If False or load fails, recomputes it.

    Returns:
        Tuple containing (train_dataset, val_dataset, test_dataset).
    """
    # 1. Load Metadata
    # Metadata files are pre-generated in ./metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 2. Prepare ROI Cache
    # Combine all dataframes to ensure the cache covers every subject in one pass.
    # The get_roi_cache function handles the logic of checking disk vs computing.
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    roi_cache = get_roi_cache(combined_df, load_cached_data=load_cached_data)

    # 3. Create Datasets
    train_dataset = BrainTumorDataset(train_df, roi_cache, phase="train")
    val_dataset = BrainTumorDataset(val_df, roi_cache, phase="val")
    test_dataset = BrainTumorDataset(test_df, roi_cache, phase="test")

    return train_dataset, val_dataset, test_dataset

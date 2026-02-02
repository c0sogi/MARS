import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VALIDATION_METADATA_PATH,
    TEST_METADATA_PATH,
    IMAGE_SIZE,
    ASH_BAND_IDS,
    DIFF_BAND_IDS,
)

# Constants for Ash Composite Normalization
# Based on standard GOES-16 Ash RGB recipes
# Red: T15 - T14, Range [-4, 2] (approx)
# Green: T14 - T11, Range [-4, 5] (approx)
# Blue: T14, Range [243, 303] (approx)
ASH_MIN = np.array([-4.0, -4.0, 243.0])
ASH_MAX = np.array([2.0, 5.0, 303.0])


def normalize_range(data, min_val, max_val):
    """
    Normalizes data to [0, 1] based on provided min/max bounds.
    Clips values outside the range.
    """
    return np.clip((data - min_val) / (max_val - min_val), 0, 1)


def get_transforms(stage: str):
    """
    Returns the Albumentations transform pipeline for the given stage.

    Args:
        stage (str): 'train', 'validation', or 'test'.
    """
    if stage == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # For validation/test, just convert to tensor
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class ContrailDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        load_cached_data=True,
        max_samples=None,
    ):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use disk caching for processed inputs.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Select Metadata File
        if split == "train":
            self.metadata_path = TRAIN_METADATA_PATH
        elif split == "validation":
            self.metadata_path = VALIDATION_METADATA_PATH
        elif split == "test":
            self.metadata_path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Metadata
        self.df = pd.read_csv(self.metadata_path)

        # Convert record_id to string to ensure consistency
        self.df["record_id"] = self.df["record_id"].astype(str)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # Setup Cache Directory for this split
        self.split_cache_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(self.split_cache_dir, exist_ok=True)

        # Pre-compute spatial coordinates (static for all images of same size)
        # Normalized coordinates [0, 1]
        y_coords = np.linspace(0, 1, IMAGE_SIZE)
        x_coords = np.linspace(0, 1, IMAGE_SIZE)
        self.mesh_y, self.mesh_x = np.meshgrid(y_coords, x_coords, indexing="ij")

    def __len__(self):
        return len(self.df)

    def _load_band(self, path):
        """Loads a single band .npy file."""
        full_path = os.path.join(INPUT_DIR, path)
        return np.load(full_path)

    def _process_record(self, row):
        """
        Generates the 8-channel input tensor for a single record.

        Channels:
        0: Ash R (Normalized)
        1: Ash G (Normalized)
        2: Ash B (Normalized)
        3: Diff Band 11 (Raw)
        4: Diff Band 14 (Raw)
        5: Diff Band 15 (Raw)
        6: Coord Y (Normalized)
        7: Coord X (Normalized)
        """
        record_id = str(row["record_id"])
        cache_path = os.path.join(self.split_cache_dir, f"{record_id}.npy")

        # 1. Check Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                # If load fails, fall back to computation
                pass

        # 2. Load Raw Bands
        # We need bands 11, 14, 15 for both Ash and Diff
        # Shape: H x W x T (T=8)
        # T=4 is the labeled frame (5th image)
        # T=3 is the previous frame (4th image)

        band_11 = self._load_band(row["band_11"])
        band_14 = self._load_band(row["band_14"])
        band_15 = self._load_band(row["band_15"])

        # Extract time steps
        t_curr = 4
        t_prev = 3

        b11_curr = band_11[..., t_curr]
        b14_curr = band_14[..., t_curr]
        b15_curr = band_15[..., t_curr]

        b11_prev = band_11[..., t_prev]
        b14_prev = band_14[..., t_prev]
        b15_prev = band_15[..., t_prev]

        # 3. Compute Ash Composite (Channels 0-2)
        # Red: T15 - T14
        r = b15_curr - b14_curr
        # Green: T14 - T11
        g = b14_curr - b11_curr
        # Blue: T14
        b = b14_curr

        r_norm = normalize_range(r, ASH_MIN[0], ASH_MAX[0])
        g_norm = normalize_range(g, ASH_MIN[1], ASH_MAX[1])
        b_norm = normalize_range(b, ASH_MIN[2], ASH_MAX[2])

        # 4. Compute Temporal Differences (Channels 3-5)
        # Raw differences: Current - Previous
        diff_11 = b11_curr - b11_prev
        diff_14 = b14_curr - b14_prev
        diff_15 = b15_curr - b15_prev

        # 5. Spatial Coordinates (Channels 6-7)
        # Use pre-computed meshgrids
        coord_y = self.mesh_y
        coord_x = self.mesh_x

        # 6. Stack Channels
        # Shape: (H, W, 8)
        img = np.stack(
            [
                r_norm,
                g_norm,
                b_norm,
                diff_11,
                diff_14,
                diff_15,
                coord_y,
                coord_x,
            ],
            axis=-1,
        ).astype(np.float32)

        # 7. Save to Cache
        if self.load_cached_data:
            np.save(cache_path, img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Input Image
        # Shape: (H, W, 8)
        image = self._process_record(row)

        # Load Mask (if available)
        mask = None
        if "human_pixel_masks" in row and pd.notna(row["human_pixel_masks"]):
            mask_path = os.path.join(INPUT_DIR, row["human_pixel_masks"])
            # Shape: (H, W, 1)
            mask = np.load(mask_path).astype(np.float32)
        else:
            # Create dummy mask for test set
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)

        # Apply Transforms
        if self.transform:
            # Albumentations expects (H, W, C)
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Albumentations ToTensorV2 converts image to (C, H, W)
            # But mask might still be (H, W) or (H, W, 1) depending on transform
            # Ensure mask is (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                # If it wasn't transposed by ToTensorV2 (sometimes mask isn't if not target)
                mask = mask.permute(2, 0, 1)

        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1)  # (8, H, W)
            mask = torch.from_numpy(mask).permute(2, 0, 1)  # (1, H, W)

        return image, mask

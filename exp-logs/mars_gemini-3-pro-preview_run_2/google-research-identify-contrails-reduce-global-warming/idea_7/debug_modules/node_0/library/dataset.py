import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.

    Loads GOES-16 satellite imagery bands, computes Ash False Color Composites,
    and generates a 6-channel input tensor (Current Frame Ash + Temporal Difference).
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: A.Compose = None,
        debug: bool = False,
        return_record_id: bool = False,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing file paths and record IDs.
            transform (A.Compose): Albumentations transform pipeline.
            debug (bool): If True, limits dataset to Config.DEBUG_SAMPLE_SIZE.
            return_record_id (bool): If True, returns (image, record_id) or (image, mask, record_id).
                                     Useful for test set inference.
        """
        self.metadata = metadata_df.copy()
        self.transform = transform
        self.return_record_id = return_record_id

        # Handle Debug Mode
        if debug:
            self.metadata = self.metadata.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Pre-extract paths to lists for faster access in __getitem__
        self.records = self.metadata["record_id"].astype(str).tolist()
        self.band_11_paths = self.metadata["band_11"].tolist()
        self.band_14_paths = self.metadata["band_14"].tolist()
        self.band_15_paths = self.metadata["band_15"].tolist()

        # Check for masks
        self.has_masks = "human_pixel_masks" in self.metadata.columns
        if self.has_masks:
            self.mask_paths = self.metadata["human_pixel_masks"].tolist()

    def __len__(self):
        return len(self.records)

    def normalize(self, data, min_val, max_val):
        """
        Linearly normalizes data to [0, 1] based on provided bounds.
        Clips values outside the range.
        """
        return np.clip((data - min_val) / (max_val - min_val), 0, 1)

    def get_ash_vector(self, b11, b14, b15):
        """
        Computes the Ash False Color Composite.

        Args:
            b11, b14, b15: 2D numpy arrays of brightness temperatures.

        Returns:
            np.ndarray: H x W x 3 array (Red, Green, Blue).
        """
        # Ash Color Recipe
        # Red: Band 15 - Band 14 (Range: -4 to 2 K)
        # Green: Band 14 - Band 11 (Range: -4 to 5 K)
        # Blue: Band 14 (Range: 243 to 303 K)

        r = self.normalize(b15 - b14, -4, 2)
        g = self.normalize(b14 - b11, -4, 5)
        b = self.normalize(b14, 243, 303)

        return np.stack([r, g, b], axis=-1)

    def __getitem__(self, idx):
        # 1. Load Raw Bands
        # File content shape: H x W x T (where T=8 usually)
        path_11 = os.path.join(Config.INPUT_DIR, self.band_11_paths[idx])
        path_14 = os.path.join(Config.INPUT_DIR, self.band_14_paths[idx])
        path_15 = os.path.join(Config.INPUT_DIR, self.band_15_paths[idx])

        try:
            # Load data
            # Note: Using mmap_mode='r' could be faster for large files, but these are small (~2MB)
            b11 = np.load(path_11).astype(np.float32)
            b14 = np.load(path_14).astype(np.float32)
            b15 = np.load(path_15).astype(np.float32)
        except Exception as e:
            # Fallback for corrupt files (though validation script showed 0 missing)
            print(f"Error loading {self.records[idx]}: {e}")
            # Return zeros of expected shape (256, 256, 6)
            dummy_img = torch.zeros((6, Config.IMG_SIZE, Config.IMG_SIZE))
            dummy_mask = torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))
            if self.return_record_id:
                return dummy_img, dummy_mask, self.records[idx]
            return dummy_img, dummy_mask

        # 2. Extract Time Steps
        # n_times_before = 4.
        # Index 4 is the labeled frame (current).
        # Index 3 is the previous frame (-10 mins).
        idx_curr = 4
        idx_prev = 3

        # 3. Compute Features
        # Current Frame Ash
        ash_curr = self.get_ash_vector(
            b11[..., idx_curr], b14[..., idx_curr], b15[..., idx_curr]
        )

        # Previous Frame Ash
        ash_prev = self.get_ash_vector(
            b11[..., idx_prev], b14[..., idx_prev], b15[..., idx_prev]
        )

        # Temporal Difference (Current - Prev)
        # This highlights moving linear features
        ash_diff = ash_curr - ash_prev

        # Concatenate to 6 channels: [Ash_R, Ash_G, Ash_B, Diff_R, Diff_G, Diff_B]
        image = np.concatenate([ash_curr, ash_diff], axis=-1)  # Shape: (H, W, 6)

        # 4. Load Mask (if available)
        mask = None
        if self.has_masks:
            mask_path = os.path.join(Config.INPUT_DIR, self.mask_paths[idx])
            mask = np.load(mask_path).astype(np.float32)  # Shape: (H, W, 1)

        # 5. Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Manual ToTensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            if mask is not None:
                mask = torch.from_numpy(mask).permute(2, 0, 1)  # (C, H, W)

        # Ensure correct types
        image = image.float()

        if mask is not None:
            mask = mask.float()
        else:
            # Return dummy mask for test set to maintain signature consistency if needed
            mask = torch.zeros((1, image.shape[1], image.shape[2])).float()

        if self.return_record_id:
            return image, mask, self.records[idx]

        return image, mask


def get_train_transform():
    """
    Returns the Albumentations augmentation pipeline for training.
    Includes geometric transformations to improve robustness to orientation.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # ShiftScaleRotate helps with scale invariance and slight rotation
            # We avoid ElasticTransform as it distorts the linear nature of contrails
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.5,
                border_mode=0,  # Fill with 0 (Background)
                value=0,
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transform():
    """
    Returns the Albumentations pipeline for validation and testing.
    Only performs tensor conversion.
    """
    return A.Compose([ToTensorV2()])

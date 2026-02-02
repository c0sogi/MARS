import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Physical Constants for Normalization
# ==========================================
# Ash Composite Bounds (Kelvin / Kelvin Difference)
# Red: Band 15 - Band 13
ASH_RED_MIN, ASH_RED_MAX = -4.0, 2.0
# Green: Band 14 - Band 11
ASH_GREEN_MIN, ASH_GREEN_MAX = -4.0, 5.0
# Blue: Band 13
ASH_BLUE_MIN, ASH_BLUE_MAX = 243.0, 303.0

# Temporal Difference Bounds (Kelvin)
# Applied to Band 11, 14, 15 differences (t4 - t3)
TEMP_DIFF_MIN, TEMP_DIFF_MAX = -5.0, 5.0


class ContrailDataset(Dataset):
    def __init__(self, metadata_df, split="train", transform=None):
        """
        Dataset class for Contrail Segmentation.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing file paths and metadata.
            split (str): Dataset split ('train', 'validation', 'test').
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.metadata = metadata_df
        self.split = split
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Pre-compute full paths for required bands to optimize __getitem__
        # We need Bands 11, 13, 14, 15
        self.band_paths = {}
        required_bands = ["11", "13", "14", "15"]

        for b in required_bands:
            col_name = f"band_{b}"
            if col_name in self.metadata.columns:
                self.band_paths[b] = [
                    os.path.join(self.input_dir, str(p))
                    for p in self.metadata[col_name]
                ]
            else:
                raise ValueError(f"Metadata missing required column: {col_name}")

        # Handle Masks
        self.has_masks = "human_pixel_masks" in self.metadata.columns
        if self.has_masks:
            self.mask_paths = [
                os.path.join(self.input_dir, str(p))
                for p in self.metadata["human_pixel_masks"]
            ]
        else:
            self.mask_paths = None

        self.record_ids = self.metadata["record_id"].astype(str).tolist()

    def __len__(self):
        return len(self.metadata)

    def normalize(self, data, min_v, max_v):
        """
        Linearly scales data from [min_v, max_v] to [0, 1].
        Values are clipped to the range before scaling.
        """
        data = np.clip(data, min_v, max_v)
        return (data - min_v) / (max_v - min_v)

    def __getitem__(self, idx):
        """
        Generates the 6-channel input tensor and mask for a given index.

        Input Channels:
        1. Ash Red (B15 - B13) at t=4
        2. Ash Green (B14 - B11) at t=4
        3. Ash Blue (B13) at t=4
        4. Temporal Diff Band 11 (t=4 - t=3)
        5. Temporal Diff Band 14 (t=4 - t=3)
        6. Temporal Diff Band 15 (t=4 - t=3)
        """
        # Time indices: t_curr is the labeled frame (5th image, index 4)
        # t_prev is the previous frame (4th image, index 3)
        t_curr = 4
        t_prev = 3

        try:
            # Load full sequences (H, W, T)
            b11 = np.load(self.band_paths["11"][idx])
            b13 = np.load(self.band_paths["13"][idx])
            b14 = np.load(self.band_paths["14"][idx])
            b15 = np.load(self.band_paths["15"][idx])
        except Exception as e:
            # Fallback for corrupted files (though unlikely given EDA)
            raise RuntimeError(
                f"Failed to load bands for record {self.record_ids[idx]}: {e}"
            )

        # Extract frames
        b11_t4 = b11[..., t_curr]
        b13_t4 = b13[..., t_curr]
        b14_t4 = b14[..., t_curr]
        b15_t4 = b15[..., t_curr]

        b11_t3 = b11[..., t_prev]
        b14_t3 = b14[..., t_prev]
        b15_t3 = b15[..., t_prev]

        # --- Feature Engineering ---

        # 1. Ash False Color Composite (Spectral)
        # Red: T15 - T13
        ash_r = self.normalize(b15_t4 - b13_t4, ASH_RED_MIN, ASH_RED_MAX)
        # Green: T14 - T11
        ash_g = self.normalize(b14_t4 - b11_t4, ASH_GREEN_MIN, ASH_GREEN_MAX)
        # Blue: T13
        ash_b = self.normalize(b13_t4, ASH_BLUE_MIN, ASH_BLUE_MAX)

        # 2. Temporal Differences (Temporal)
        # Diff 11
        diff_11 = self.normalize(b11_t4 - b11_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)
        # Diff 14
        diff_14 = self.normalize(b14_t4 - b14_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)
        # Diff 15
        diff_15 = self.normalize(b15_t4 - b15_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)

        # Stack to create (H, W, 6)
        img = np.stack(
            [ash_r, ash_g, ash_b, diff_11, diff_14, diff_15], axis=-1
        ).astype(np.float32)

        # Load Mask
        mask = None
        if self.has_masks:
            # Shape (H, W, 1)
            mask = np.load(self.mask_paths[idx]).astype(np.float32)

        # Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Ensure tensors
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            # Ensure mask is (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.shape[-1] == 1:  # HWC
                mask = mask.permute(2, 0, 1)
        else:
            # Create dummy mask for test set
            mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float32)

        return {"image": img, "mask": mask, "record_id": self.record_ids[idx]}


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline.

    Train: Affine transformations (Flip, Shift, Scale, Rotate) to preserve linearity.
    Val/Test: Normalization (ToTensorV2).
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Affine only: Avoid elastic/grid distortions that warp linear features
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant 0 padding
                    value=0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])

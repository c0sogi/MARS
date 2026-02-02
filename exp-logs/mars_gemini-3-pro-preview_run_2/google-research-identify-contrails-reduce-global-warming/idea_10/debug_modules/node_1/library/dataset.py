import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Constants & Normalization Params
# ==========================================
# Ash Color Composite Bounds (Approximate Kelvin values for GOES-16)
# Red: T15 - T14
ASH_RED_MIN = -4.0
ASH_RED_MAX = 2.0

# Green: T14 - T11
ASH_GREEN_MIN = -4.0
ASH_GREEN_MAX = 5.0

# Blue: T14
ASH_BLUE_MIN = 243.0
ASH_BLUE_MAX = 303.0

# Temporal Difference Bounds (Kelvin difference)
# Used for normalizing (T_curr - T_prev)
DIFF_MIN = -4.0
DIFF_MAX = 4.0


def normalize_range(data, min_v, max_v):
    """
    Normalizes data to [0, 1] based on provided min/max values.
    Clips values outside the range.
    """
    data = (data - min_v) / (max_v - min_v)
    return np.clip(data, 0, 1)


def get_ash_color(b11, b14, b15):
    """
    Constructs the Ash False Color Composite.
    Args:
        b11, b14, b15: 2D arrays of brightness temperatures.
    Returns:
        np.ndarray: (H, W, 3) array normalized to [0, 1].
    """
    # Red: T15 - T14
    r = normalize_range(b15 - b14, ASH_RED_MIN, ASH_RED_MAX)

    # Green: T14 - T11
    g = normalize_range(b14 - b11, ASH_GREEN_MIN, ASH_GREEN_MAX)

    # Blue: T14
    b = normalize_range(b14, ASH_BLUE_MIN, ASH_BLUE_MAX)

    return np.stack([r, g, b], axis=-1)


def get_transforms(split="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                # Affine transformations: Rotation, Scale, Shift
                # We avoid elastic/grid distortions to preserve linear features
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation/Test: Just convert to tensor
        return A.Compose([ToTensorV2(transpose_mask=True)])


class ContrailDataset(Dataset):
    """
    Dataset for Contrail Identification.
    Loads satellite bands, computes Ash composite and Temporal differences.
    """

    def __init__(self, metadata_df, split="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing file paths.
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.df = metadata_df
        self.split = split
        self.transform = transform

        # Determine indices for temporal sequence
        # Array shape is T=8.
        # n_times_before=4 -> Indices 0,1,2,3 are before.
        # Index 4 is the labeled frame.
        self.idx_current = Config.N_TIMES_BEFORE
        self.idx_prev = Config.N_TIMES_BEFORE - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # --------------------------------------
        # 1. Load Data
        # --------------------------------------
        # We need bands 11, 14, 15 for Ash Color and their differences
        # Paths are relative to Config.INPUT_DIR
        try:
            path_11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            path_14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            path_15 = os.path.join(Config.INPUT_DIR, row["band_15"])

            # Load full sequences: Shape (H, W, T)
            # Using mmap_mode='r' can be faster for large files if we only slice,
            # but here files are small (2MB), so loading full is fine.
            b11_seq = np.load(path_11)
            b14_seq = np.load(path_14)
            b15_seq = np.load(path_15)

            # Extract Current Frame (t=4) and Previous Frame (t=3)
            # Shape becomes (H, W)
            b11_curr = b11_seq[..., self.idx_current]
            b14_curr = b14_seq[..., self.idx_current]
            b15_curr = b15_seq[..., self.idx_current]

            b11_prev = b11_seq[..., self.idx_prev]
            b14_prev = b14_seq[..., self.idx_prev]
            b15_prev = b15_seq[..., self.idx_prev]

        except Exception as e:
            # Fallback for corrupted data or debugging
            print(f"Error loading data for {record_id}: {e}")
            # Return zeros
            img = torch.zeros(
                (Config.IN_CHANNELS_STAGE1, Config.IMG_SIZE, Config.IMG_SIZE)
            )
            if self.split == "test":
                return img
            return img, torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))

        # --------------------------------------
        # 2. Feature Engineering
        # --------------------------------------

        # Channels 1-3: Ash False Color Composite (Current Frame)
        ash_composite = get_ash_color(b11_curr, b14_curr, b15_curr)  # (H, W, 3)

        # Channels 4-6: Temporal Difference (Current - Previous)
        # We normalize these differences to a fixed range (e.g., -4 to 4 K)
        diff_11 = normalize_range(b11_curr - b11_prev, DIFF_MIN, DIFF_MAX)
        diff_14 = normalize_range(b14_curr - b14_prev, DIFF_MIN, DIFF_MAX)
        diff_15 = normalize_range(b15_curr - b15_prev, DIFF_MIN, DIFF_MAX)

        temporal_diff = np.stack([diff_11, diff_14, diff_15], axis=-1)  # (H, W, 3)

        # Concatenate to form 6-channel input
        # Shape: (H, W, 6)
        image = np.concatenate([ash_composite, temporal_diff], axis=-1).astype(
            np.float32
        )

        # --------------------------------------
        # 3. Load Mask (if applicable)
        # --------------------------------------
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            # Mask shape: (H, W, 1)
            mask = np.load(mask_path).astype(np.float32)

        # --------------------------------------
        # 4. Augmentation
        # --------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                # Albumentations ToTensorV2 converts mask to (H, W) or (1, H, W) depending on input
                # Ensure mask is (1, H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.shape[0] != 1:
                    # If mask was (H, W, 1) and ToTensorV2 made it (1, H, W) -> Correct
                    # But sometimes it might permute differently.
                    # Standard ToTensorV2 with HWC input -> CHW output.
                    pass
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # --------------------------------------
        # 5. Return
        # --------------------------------------
        if self.split == "test":
            return image
        else:
            return image, mask

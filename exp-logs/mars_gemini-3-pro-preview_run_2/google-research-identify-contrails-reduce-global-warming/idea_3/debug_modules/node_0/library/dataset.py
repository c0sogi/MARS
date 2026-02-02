import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Constants for Normalization
# ==========================================
# Ash Color Composite Bounds (Heuristic for GOES-16)
# Band 11 (8.4um), Band 14 (11.2um), Band 15 (12.3um)
# Red: T15 - T14
_ASH_RED_MIN, _ASH_RED_MAX = -4.0, 2.0
# Green: T14 - T11
_ASH_GREEN_MIN, _ASH_GREEN_MAX = -4.0, 5.0
# Blue: T14
_ASH_BLUE_MIN, _ASH_BLUE_MAX = 243.0, 303.0

# Temporal Difference Bounds (Heuristic)
# Assuming differences are generally small within 10 mins
_DIFF_MIN, _DIFF_MAX = -2.0, 2.0


def normalize(data, min_val, max_val):
    """
    Linearly normalizes data to [0, 1] and clips values outside the range.
    """
    data = (data - min_val) / (max_val - min_val)
    return np.clip(data, 0.0, 1.0)


def get_transforms(split):
    """
    Returns Albumentations transforms for the given split.

    Strategy:
    - Train: Horizontal/Vertical Flip, ShiftScaleRotate (Affine).
    - Validation/Test: Normalization/Tensor conversion only.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Affine transformations only; avoid elastic/grid distortions that warp lines
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant padding with 0
                    value=0,
                    mask_value=0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Detection.

    Features:
    - Loads Bands 11, 14, 15 from .npy files.
    - Constructs 6-channel input:
        - Ch 1-3: Ash Color Composite (T=4)
        - Ch 4-6: Temporal Difference (T=4 - T=3)
    - Returns: Image Tensor, Mask Tensor, Classification Label, Record ID.
    """

    def __init__(self, split="train", max_samples=None, load_cached_data=False):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            max_samples (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Placeholder for caching logic (not used for on-the-fly loading).
        """
        self.split = split

        # Select metadata file based on split
        if split == "train":
            self.meta_path = Config.TRAIN_METADATA
        elif split == "validation":
            self.meta_path = Config.VALIDATION_METADATA
        elif split == "test":
            self.meta_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        # Ensure record_id is string
        self.df["record_id"] = self.df["record_id"].astype(str)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        self.transform = get_transforms(split)

        # Time indices for the sequence
        # Sequence length T = 4 (before) + 1 (current) + 3 (after) = 8
        # Labeled frame is at index 4 (0-based)
        self.t_curr = 4
        self.t_prev = 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # ---------------------------------------------------
        # 1. Load Satellite Bands (11, 14, 15)
        # ---------------------------------------------------
        # Helper to load a band
        def load_band(band_idx):
            rel_path = row[f"band_{band_idx}"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            # Shape: (H, W, T)
            return np.load(full_path)

        band_11 = load_band(11)
        band_14 = load_band(14)
        band_15 = load_band(15)

        # ---------------------------------------------------
        # 2. Construct Ash Color Composite (Channels 1-3)
        # ---------------------------------------------------
        # Formula:
        # R = T15 - T14
        # G = T14 - T11
        # B = T14

        t11_curr = band_11[..., self.t_curr]
        t14_curr = band_14[..., self.t_curr]
        t15_curr = band_15[..., self.t_curr]

        r = normalize(t15_curr - t14_curr, _ASH_RED_MIN, _ASH_RED_MAX)
        g = normalize(t14_curr - t11_curr, _ASH_GREEN_MIN, _ASH_GREEN_MAX)
        b = normalize(t14_curr, _ASH_BLUE_MIN, _ASH_BLUE_MAX)

        ash_composite = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        # ---------------------------------------------------
        # 3. Construct Temporal Difference (Channels 4-6)
        # ---------------------------------------------------
        # Diff = Frame(t) - Frame(t-1)

        t11_prev = band_11[..., self.t_prev]
        t14_prev = band_14[..., self.t_prev]
        t15_prev = band_15[..., self.t_prev]

        d1 = normalize(t11_curr - t11_prev, _DIFF_MIN, _DIFF_MAX)
        d2 = normalize(t14_curr - t14_prev, _DIFF_MIN, _DIFF_MAX)
        d3 = normalize(t15_curr - t15_prev, _DIFF_MIN, _DIFF_MAX)

        temporal_diff = np.stack([d1, d2, d3], axis=-1)  # (H, W, 3)

        # Combine to 6 channels
        image = np.concatenate([ash_composite, temporal_diff], axis=-1)  # (H, W, 6)

        # ---------------------------------------------------
        # 4. Load Mask
        # ---------------------------------------------------
        if self.split != "test":
            mask_rel_path = row["human_pixel_masks"]
            mask_path = os.path.join(Config.INPUT_DIR, mask_rel_path)
            mask = np.load(mask_path)  # (H, W, 1)
        else:
            # Dummy mask for test set
            h, w = image.shape[:2]
            mask = np.zeros((h, w, 1), dtype=np.float32)

        # ---------------------------------------------------
        # 5. Augmentations
        # ---------------------------------------------------
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Tensor (6, H, W)
            mask = augmented["mask"]  # Tensor (1, H, W) or (H, W)

        # Ensure mask is (1, H, W) float tensor
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            # If ToTensorV2 permuted to (1, H, W), it's fine.
            # If it's still (H, W, 1) (unlikely with ToTensorV2 but possible if config changes), permute.
            if mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1)

        mask = mask.float()
        image = image.float()

        # ---------------------------------------------------
        # 6. Classification Label
        # ---------------------------------------------------
        # 1.0 if contrail exists (any pixel > 0), 0.0 otherwise.
        # This is used for the auxiliary classification head.
        label = 1.0 if mask.sum() > 0 else 0.0
        label = torch.tensor(label, dtype=torch.float32)

        return image, mask, label, record_id

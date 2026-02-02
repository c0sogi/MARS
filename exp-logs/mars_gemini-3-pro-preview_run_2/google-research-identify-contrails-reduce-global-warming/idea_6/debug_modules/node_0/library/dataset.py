import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def normalize_range(data, low, high):
    """
    Normalizes data from [low, high] to [0, 1].
    Clips values outside the range.
    """
    return (np.clip(data, low, high) - low) / (high - low)


def get_ash_color(band11, band14, band15):
    """
    Computes the Ash False Color Composite.

    Args:
        band11, band14, band15: 2D numpy arrays of brightness temperatures (Kelvin).

    Returns:
        np.ndarray: HxWx3 array normalized to [0, 1].
    """
    # Red: T15 - T14
    r = band15 - band14
    r = normalize_range(r, -4, 2)

    # Green: T14 - T11
    g = band14 - band11
    g = normalize_range(g, -4, 5)

    # Blue: T14
    b = band14
    b = normalize_range(b, 243, 303)

    return np.stack([r, g, b], axis=-1)


# ==========================================
# Dataset Class
# ==========================================


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.
    Loads satellite bands, computes Ash composite and temporal differences,
    and applies affine augmentations.
    """

    def __init__(self, split="train", transform=None, debug=False):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.transform = transform
        self.cfg = Config

        # Determine metadata file path
        if split == "train":
            self.metadata_path = self.cfg.TRAIN_METADATA_PATH
        elif split == "validation":
            self.metadata_path = self.cfg.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = self.cfg.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debug mode: sample subset
        if debug or self.cfg.DEBUG:
            print(
                f"DEBUG MODE: Sampling {self.cfg.DEBUG_SAMPLE_SIZE} records from {split}."
            )
            self.df = self.df.sample(
                n=min(len(self.df), self.cfg.DEBUG_SAMPLE_SIZE),
                random_state=self.cfg.SEED,
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # Define frame indices
        # t=4 is the labeled frame (index 4)
        # t=3 is the previous frame (index 3)
        idx_t4 = 4
        idx_t3 = 3

        try:
            # Load Bands 11, 14, 15
            # Files are H x W x T
            p11 = os.path.join(self.cfg.INPUT_DIR, row["band_11"])
            p14 = os.path.join(self.cfg.INPUT_DIR, row["band_14"])
            p15 = os.path.join(self.cfg.INPUT_DIR, row["band_15"])

            b11_all = np.load(p11)
            b14_all = np.load(p14)
            b15_all = np.load(p15)

            # Extract specific time steps
            b11_t4 = b11_all[..., idx_t4]
            b14_t4 = b14_all[..., idx_t4]
            b15_t4 = b15_all[..., idx_t4]

            b11_t3 = b11_all[..., idx_t3]
            b14_t3 = b14_all[..., idx_t3]
            b15_t3 = b15_all[..., idx_t3]

            # 1. Compute Ash Color (Channels 1-3)
            ash_composite = get_ash_color(b11_t4, b14_t4, b15_t4)

            # 2. Compute Temporal Difference (Channels 4-6)
            # We use raw band differences.
            # Normalization: Diffs are small, usually within +/- 5K.
            # We scale them to be roughly in [-1, 1] range to aid convergence.
            # Divisor 2.0 puts a 2K diff at 1.0.
            diff_11 = (b11_t4 - b11_t3) / 2.0
            diff_14 = (b14_t4 - b14_t3) / 2.0
            diff_15 = (b15_t4 - b15_t3) / 2.0

            temporal_diff = np.stack([diff_11, diff_14, diff_15], axis=-1)

            # Concatenate to form 6-channel input
            # Shape: H x W x 6
            image = np.concatenate([ash_composite, temporal_diff], axis=-1).astype(
                np.float32
            )

            # Load Mask (if available)
            mask = None
            if self.split in ["train", "validation"]:
                mask_path = os.path.join(self.cfg.INPUT_DIR, row["human_pixel_masks"])
                # Shape: H x W x 1
                mask = np.load(mask_path).astype(np.float32)

            # Apply Augmentations
            if self.transform:
                if mask is not None:
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented["image"]
                    mask = augmented["mask"]
                    # Ensure mask is channel-first if ToTensorV2 didn't handle it (it usually does)
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    elif mask.shape[-1] == 1 and not isinstance(mask, torch.Tensor):
                        mask = mask.transpose(2, 0, 1)
                else:
                    augmented = self.transform(image=image)
                    image = augmented["image"]

            # If no transform (shouldn't happen with get_transforms), convert manually
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1))
                if mask is not None and not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask.transpose(2, 0, 1))

            # Return dict
            result = {"image": image, "record_id": record_id}
            if mask is not None:
                result["mask"] = mask

            return result

        except Exception as e:
            print(f"Error loading record {record_id}: {e}")
            # Return a zero tensor in case of error to prevent crash,
            # but in a competition usually better to raise or skip.
            # Here we raise to identify issues early.
            raise e


# ==========================================
# Augmentation Factory
# ==========================================


def get_transforms(split="train", cfg=Config):
    """
    Returns the Albumentations transform pipeline for the given split.

    Args:
        split (str): 'train', 'validation', or 'test'.
        cfg (Config): Configuration object.

    Returns:
        albumentations.Compose: Transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                # Affine Transformations only (No Elastic/Grid/Optical)
                A.ShiftScaleRotate(
                    shift_limit=cfg.AUG_SHIFT_LIMIT,
                    scale_limit=cfg.AUG_SCALE_LIMIT,
                    rotate_limit=cfg.AUG_ROTATE_LIMIT,
                    p=cfg.AUG_PROB,
                    border_mode=0,  # Constant padding with 0
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([ToTensorV2()])

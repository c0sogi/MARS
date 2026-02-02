import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config

# ==========================================
# Constants for Ash Color Scheme
# ==========================================
# Bounds derived from domain knowledge for GOES-16 ABI
_ASH_R_BOUNDS = (-6.7, 2.6)  # Band 15 - Band 14
_ASH_G_BOUNDS = (-6.0, 6.3)  # Band 14 - Band 11
_ASH_B_BOUNDS = (243, 303)  # Band 14


def normalize_range(data, bounds):
    """
    Normalizes data from [min, max] to [0, 1].
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_type (str): 'train', 'validation', or 'test'.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                    value=0,
                ),
                # transpose_mask=True ensures (H, W, 1) -> (1, H, W)
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation and Test: No geometric augmentations
        return A.Compose([ToTensorV2(transpose_mask=True)])


class ContrailDataset(Dataset):
    """
    Dataset class for loading Satellite Imagery for Contrail Detection.

    Features:
    - Loads metadata from CSV.
    - Loads specific bands (11, 14, 15) from .npy files.
    - Generates Dual-Stream Inputs:
        1. Stream A: Ash False Color Composite (Static, t=4)
        2. Stream B: Raw Band Differences (Dynamic, t=4 - t=3)
    - Loads Ground Truth Masks (Consensus).
    - Applies Augmentations.
    """

    def __init__(self, metadata_path, split="train", transform=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Transformations to apply.
            debug (bool): If True, subsamples the dataset for debugging.
        """
        self.split = split
        self.transform = transform
        self.input_root = Config.INPUT_ROOT

        # Load metadata
        if not os.path.exists(metadata_path):
            # If test metadata is missing (e.g. during initial setup), handle gracefully
            if split == "test":
                print(
                    f"Warning: Metadata file not found at {metadata_path}. Creating empty DataFrame."
                )
                self.df = pd.DataFrame(columns=["record_id"])
            else:
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        else:
            self.df = pd.read_csv(metadata_path)

        # Ensure record_id is treated as string
        if not self.df.empty:
            self.df["record_id"] = self.df["record_id"].astype(str)

        # Debug mode: subsample
        if debug and not self.df.empty:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLES), random_state=Config.SEED
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # -----------------------------------------------------------------
        # 1. Load Satellite Bands
        # -----------------------------------------------------------------
        # We need bands 11, 14, 15.
        # The metadata contains relative paths, e.g., "train/123.../band_11.npy"

        def load_band(band_num):
            rel_path = row[f"band_{band_num:02d}"]
            full_path = os.path.join(self.input_root, rel_path)
            return np.load(full_path)  # Shape: H x W x T (usually 256x256x8)

        b11 = load_band(11)
        b14 = load_band(14)
        b15 = load_band(15)

        # -----------------------------------------------------------------
        # 2. Input Engineering
        # -----------------------------------------------------------------
        # Time index for the labeled frame is 4 (N_TIMES_BEFORE)
        t_curr = Config.N_TIMES_BEFORE
        t_prev = t_curr - 1

        # Extract frames for Static Stream (t=4)
        t11_curr = b11[..., t_curr]
        t14_curr = b14[..., t_curr]
        t15_curr = b15[..., t_curr]

        # Extract frames for Dynamic Stream (t=3)
        t11_prev = b11[..., t_prev]
        t14_prev = b14[..., t_prev]
        t15_prev = b15[..., t_prev]

        # --- Stream A: Ash False Color Composite ---
        # R = T15 - T14
        # G = T14 - T11
        # B = T14
        # Normalize to [0, 1]
        r_ash = normalize_range(t15_curr - t14_curr, _ASH_R_BOUNDS)
        g_ash = normalize_range(t14_curr - t11_curr, _ASH_G_BOUNDS)
        b_ash = normalize_range(t14_curr, _ASH_B_BOUNDS)

        ash_composite = np.stack([r_ash, g_ash, b_ash], axis=-1)
        ash_composite = np.clip(ash_composite, 0, 1)

        # --- Stream B: Raw Band Differences ---
        # Delta = T_curr - T_prev
        # We preserve raw values (Kelvin differences) as requested
        diff_11 = t11_curr - t11_prev
        diff_14 = t14_curr - t14_prev
        diff_15 = t15_curr - t15_prev

        diff_composite = np.stack([diff_11, diff_14, diff_15], axis=-1)

        # Combine Streams: Channel dimension is last for now (H, W, C)
        # Total Channels: 3 (Ash) + 3 (Diff) = 6
        img = np.concatenate([ash_composite, diff_composite], axis=-1).astype(
            np.float32
        )

        # -----------------------------------------------------------------
        # 3. Load Masks (Train/Valid only)
        # -----------------------------------------------------------------
        mask = None
        if self.split != "test":
            mask_rel_path = row["human_pixel_masks"]
            mask_path = os.path.join(self.input_root, mask_rel_path)

            # Load mask: H x W x 1
            mask = np.load(mask_path).astype(np.float32)

        # -----------------------------------------------------------------
        # 4. Augmentations
        # -----------------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Ensure mask is (1, H, W) if it exists
        # ToTensorV2(transpose_mask=True) should handle (H, W, 1) -> (1, H, W)
        # But if mask was (H, W) initially, it becomes (H, W) or (1, H, W) depending on internal logic.
        # We enforce 3 dimensions (C, H, W)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        # Return
        if self.split == "test":
            # Return dummy mask for test
            return img, torch.zeros((1, img.shape[1], img.shape[2])), record_id
        else:
            return img, mask, record_id

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==============================
# Constants
# ==============================

# Ash False-Color Recipe Bounds
# Based on standard GOES-16 Ash RGB recipe adapted for this dataset
# Red: T15 - T14
_TDIFF_15_14_MIN = -4.0
_TDIFF_15_14_MAX = 2.0

# Green: T14 - T11
_TDIFF_14_11_MIN = -4.0
_TDIFF_14_11_MAX = 5.0

# Blue: T14 (Brightness Temperature)
_T14_MIN = 243.0
_T14_MAX = 303.0

# ImageNet Normalization Stats
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================
# Helper Functions
# ==============================


def normalize_range(data, min_val, max_val):
    """Normalizes data to [0, 1] range based on min/max bounds."""
    return (data - min_val) / (max_val - min_val)


def get_ash_composite(band11, band14, band15):
    """
    Constructs the Ash False-Color Composite from GOES-16 bands.

    Args:
        band11 (np.array): Band 11 brightness temperatures (H, W).
        band14 (np.array): Band 14 brightness temperatures (H, W).
        band15 (np.array): Band 15 brightness temperatures (H, W).

    Returns:
        np.array: Ash composite image of shape (H, W, 3) in range [0, 1].
    """
    # Red channel: T15 - T14
    r = normalize_range(band15 - band14, _TDIFF_15_14_MIN, _TDIFF_15_14_MAX)

    # Green channel: T14 - T11
    g = normalize_range(band14 - band11, _TDIFF_14_11_MIN, _TDIFF_14_11_MAX)

    # Blue channel: T14
    b = normalize_range(band14, _T14_MIN, _T14_MAX)

    # Stack and clip
    ash = np.stack([r, g, b], axis=-1)
    return np.clip(ash, 0, 1)


def apply_imagenet_norm(img):
    """
    Applies ImageNet normalization to a (H, W, 3) image.

    Args:
        img (np.array): Image in range [0, 1].

    Returns:
        np.array: Normalized image.
    """
    return (img - _IMAGENET_MEAN) / _IMAGENET_STD


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: Transform pipeline.
    """
    if split == "train":
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
                ),
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()])


# ==============================
# Dataset Class
# ==============================


class ContrailDataset(Dataset):
    def __init__(
        self, metadata_path, split="train", transform=None, load_cached_data=False
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'validation', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Placeholder for caching logic (not used for lazy loading).
        """
        self.split = split
        self.transform = transform or get_transforms(split)

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Pre-compute input directory path for joining
        self.input_root = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Raw Bands
        # We need bands 11, 14, 15.
        # The CSV contains relative paths like "train/123.../band_11.npy"
        # We assume the file structure allows loading the full temporal sequence.

        try:
            # Load full temporal arrays (H, W, T)
            # Paths in CSV are relative to input dir
            b11_path = os.path.join(self.input_root, row["band_11"])
            b14_path = os.path.join(self.input_root, row["band_14"])
            b15_path = os.path.join(self.input_root, row["band_15"])

            # Load data. Memory mapping could be used, but files are small enough for standard load.
            b11_all = np.load(b11_path)
            b14_all = np.load(b14_path)
            b15_all = np.load(b15_path)

        except Exception as e:
            print(f"Error loading bands for record {record_id}: {e}")
            # Return zero tensors in case of read failure to prevent crash
            return torch.zeros(
                (Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
            ), torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))

        # 2. Extract Temporal Context
        # Target frame is at index t. We need t, t-1, t-2.
        t = Config.TARGET_FRAME_IDX
        indices = [t, t - 1, t - 2]

        # 3. Compute Ash and Normalize for each timestep
        ash_frames = []

        for t_idx in indices:
            # Extract specific frame (H, W)
            # Handle potential boundary issues if t_idx < 0 (though dataset spec says t=4)
            if t_idx < 0:
                # Fallback: duplicate the first frame
                idx_to_use = 0
            else:
                idx_to_use = t_idx

            b11 = b11_all[..., idx_to_use]
            b14 = b14_all[..., idx_to_use]
            b15 = b15_all[..., idx_to_use]

            # Compute Ash (H, W, 3) in [0, 1]
            ash = get_ash_composite(b11, b14, b15)

            # Apply ImageNet Normalization
            ash_norm = apply_imagenet_norm(ash)

            ash_frames.append(ash_norm)

        # Unpack frames
        ash_t = ash_frames[0]  # Frame at t
        ash_tm1 = ash_frames[1]  # Frame at t-1
        ash_tm2 = ash_frames[2]  # Frame at t-2

        # 4. Construct Multi-Order Input Tensor
        # Channels 1-3: Ash at t
        # Channels 4-6: Velocity (Ash_t - Ash_tm1)
        # Channels 7-9: Acceleration (Ash_tm1 - Ash_tm2)

        diff_velocity = ash_t - ash_tm1
        diff_acceleration = ash_tm1 - ash_tm2

        # Concatenate along channel axis (last axis for numpy: H, W, C)
        # Result shape: (H, W, 9)
        input_img = np.concatenate([ash_t, diff_velocity, diff_acceleration], axis=-1)

        # 5. Load Mask (if applicable)
        mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.float32)

        if self.split in ["train", "validation"]:
            mask_path_rel = row.get("human_pixel_masks", None)
            if mask_path_rel and isinstance(mask_path_rel, str):
                full_mask_path = os.path.join(self.input_root, mask_path_rel)
                if os.path.exists(full_mask_path):
                    # Load mask (H, W, 1)
                    mask_raw = np.load(full_mask_path)
                    # Squeeze to (H, W)
                    mask = mask_raw.squeeze(-1).astype(np.float32)

        # 6. Apply Augmentations
        if self.transform:
            # Albumentations expects image as (H, W, C)
            transformed = self.transform(image=input_img, mask=mask)
            input_tensor = transformed["image"]
            mask_tensor = transformed["mask"]

            # Ensure mask has channel dimension (1, H, W)
            # Albumentations ToTensorV2 converts mask to (H, W) if it was 2D input
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            # Manual conversion if no transform provided
            input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return input_tensor, mask_tensor

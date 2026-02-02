import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Physical Constants for Normalization
# =========================================================================
# Ash False Color Composite Bounds (Brightness Temperature Differences in Kelvin)
# Red: Band 15 - Band 14 (Optical Depth proxy)
T_DIFF_15_14_MIN = -4.0
T_DIFF_15_14_MAX = 2.0

# Green: Band 14 - Band 11 (Particle Phase proxy)
T_DIFF_14_11_MIN = -4.0
T_DIFF_14_11_MAX = 5.0

# Blue: Band 14 (Temperature proxy)
T_14_MIN = 243.0
T_14_MAX = 303.0

# Temporal Difference Normalization Bounds
# We normalize the temporal difference (T4 - T3) to a similar [0, 1] range.
# Differences are typically small (< 5K), so we map [-5, 5] to [0, 1].
TEMP_DIFF_MIN = -5.0
TEMP_DIFF_MAX = 5.0


def normalize_range(data, min_v, max_v):
    """
    Linearly normalizes data to [0, 1] based on provided min/max bounds.
    Values are clipped to the range.
    """
    return np.clip((data - min_v) / (max_v - min_v), 0, 1)


class ContrailDataset(Dataset):
    """
    Dataset class for loading Satellite Imagery and Contrail Masks.

    Features:
    - Loads Bands 11, 14, 15.
    - Constructs 6-channel input:
        1. Ash False Color Composite (3 channels)
        2. Temporal Difference (3 channels: Frame t=4 - Frame t=3)
    - Applies strict affine augmentations (Rotation, Scale, Shift, Flip).
    """

    def __init__(self, metadata_df, transform=None, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing file paths.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'validation', or 'test'.
        """
        self.df = metadata_df
        self.transform = transform
        self.mode = mode

        # Pre-compute full file paths
        self.records = self.df["record_id"].astype(str).tolist()

        # Helper to join paths
        def get_paths(col_name):
            return [os.path.join(Config.INPUT_DIR, p) for p in self.df[col_name]]

        self.band_11_paths = get_paths("band_11")
        self.band_14_paths = get_paths("band_14")
        self.band_15_paths = get_paths("band_15")

        if self.mode != "test":
            self.mask_paths = get_paths("human_pixel_masks")
        else:
            self.mask_paths = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # 1. Load Raw Data
        try:
            # Shape: (H, W, T) where T=8
            b11 = np.load(self.band_11_paths[idx])
            b14 = np.load(self.band_14_paths[idx])
            b15 = np.load(self.band_15_paths[idx])
        except Exception as e:
            # Fallback for read errors (safety mechanism)
            print(f"Error loading record {self.records[idx]}: {e}")
            img = torch.zeros(
                (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=torch.float32,
            )
            if self.mode != "test":
                mask = torch.zeros(
                    (1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
                )
                return img, mask
            return img, self.records[idx]

        # 2. Extract Temporal Frames
        # t=4 is the labeled frame, t=3 is the previous frame
        idx_t4 = 4
        idx_t3 = 3

        b11_t4 = b11[..., idx_t4]
        b14_t4 = b14[..., idx_t4]
        b15_t4 = b15[..., idx_t4]

        b11_t3 = b11[..., idx_t3]
        b14_t3 = b14[..., idx_t3]
        b15_t3 = b15[..., idx_t3]

        # 3. Feature Engineering

        # --- Ash False Color Composite (Channels 0-2) ---
        # Red: Optical Depth (T15 - T14)
        r = normalize_range(b15_t4 - b14_t4, T_DIFF_15_14_MIN, T_DIFF_15_14_MAX)
        # Green: Particle Phase (T14 - T11)
        g = normalize_range(b14_t4 - b11_t4, T_DIFF_14_11_MIN, T_DIFF_14_11_MAX)
        # Blue: Temperature (T14)
        b = normalize_range(b14_t4, T_14_MIN, T_14_MAX)

        # --- Temporal Difference (Channels 3-5) ---
        # Difference of the raw bands used in Ash, normalized to [0, 1]
        diff_11 = normalize_range(b11_t4 - b11_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)
        diff_14 = normalize_range(b14_t4 - b14_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)
        diff_15 = normalize_range(b15_t4 - b15_t3, TEMP_DIFF_MIN, TEMP_DIFF_MAX)

        # Stack to create 6-channel image: (H, W, 6)
        img = np.stack([r, g, b, diff_11, diff_14, diff_15], axis=-1).astype(np.float32)

        # 4. Load Mask (if available)
        mask = None
        if self.mode != "test":
            # Shape: (H, W, 1)
            mask = np.load(self.mask_paths[idx]).astype(np.float32)

        # 5. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            t = ToTensorV2()
            if mask is not None:
                augmented = t(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = t(image=img)
                img = augmented["image"]

        # Ensure mask is channel-first (1, H, W)
        if mask is not None:
            # Explicitly permute HWC -> CHW if necessary
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)

        if self.mode != "test":
            return img, mask
        else:
            return img, self.records[idx]


def get_transforms(mode="train"):
    """
    Defines the augmentation pipeline.

    Strategy:
    - Train: Horizontal/Vertical Flips, ShiftScaleRotate.
      Explicitly avoids elastic/grid distortions to preserve linear morphology.
    - Val/Test: Normalization/Tensor conversion only.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=180,  # Isotropic nature allows full rotation
                    p=0.5,
                    border_mode=0,  # Constant 0 padding
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_train_val_loaders(load_cached_data=False):
    """
    Constructs DataLoaders for training and validation sets.

    Args:
        load_cached_data (bool): Placeholder for compatibility.
                                 Dataframes are loaded directly from CSVs.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debugging / Rapid Prototyping
    if Config.MAX_TRAIN_SAMPLES is not None:
        train_df = train_df.sample(
            n=min(len(train_df), Config.MAX_TRAIN_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)

    if Config.MAX_VAL_SAMPLES is not None:
        val_df = val_df.sample(
            n=min(len(val_df), Config.MAX_VAL_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = ContrailDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_dataset = ContrailDataset(
        val_df, transform=get_transforms("validation"), mode="validation"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Constructs DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    dataset = ContrailDataset(test_df, transform=get_transforms("test"), mode="test")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader

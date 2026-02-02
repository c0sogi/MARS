import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed

# --- Constants for Ash Color Scheme Normalization ---
# Based on standard heuristics for GOES-16 Contrail Detection
# Red: T15 - T14
_ASH_R_MIN, _ASH_R_MAX = -6.7, 2.6
# Green: T14 - T11
_ASH_G_MIN, _ASH_G_MAX = -6.0, 6.0
# Blue: T14
_ASH_B_MIN, _ASH_B_MAX = 240.0, 300.0


def normalize_range(data, vmin, vmax):
    """
    Normalizes data to [0, 1] based on provided min/max bounds.
    Clips values outside the range.
    """
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)


def get_ash_color(t11, t14, t15):
    """
    Constructs the Ash False-Color Composite from brightness temperatures.

    Args:
        t11 (np.ndarray): Band 11 brightness temperatures.
        t14 (np.ndarray): Band 14 brightness temperatures.
        t15 (np.ndarray): Band 15 brightness temperatures.

    Returns:
        np.ndarray: Normalized Ash composite of shape (H, W, 3).
    """
    # Red component: Optical depth proxy (T15 - T14)
    r = normalize_range(t15 - t14, _ASH_R_MIN, _ASH_R_MAX)

    # Green component: Particle size/phase proxy (T14 - T11)
    g = normalize_range(t14 - t11, _ASH_G_MIN, _ASH_G_MAX)

    # Blue component: Temperature (T14)
    b = normalize_range(t14, _ASH_B_MIN, _ASH_B_MAX)

    return np.stack([r, g, b], axis=-1)


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.
    Loads satellite bands, constructs physics-informed features, and applies augmentations.
    """

    def __init__(self, split="train", transform=None, debug=Config.DEBUG):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.transform = transform
        self.debug = debug

        # Load Metadata
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            self.metadata_path = Config.VALIDATION_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debug Mode: Sample a small subset
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), 50), random_state=Config.SEED
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # --- Load Bands ---
        # We need Band 11, 14, 15 for Ash Color
        # Files are (H, W, T) where T=8.
        # Current frame (t) is index 4. Previous frame (t-1) is index 3.

        try:
            # Load full temporal sequences
            band_11 = np.load(os.path.join(Config.INPUT_ROOT, row["band_11"]))
            band_14 = np.load(os.path.join(Config.INPUT_ROOT, row["band_14"]))
            band_15 = np.load(os.path.join(Config.INPUT_ROOT, row["band_15"]))

            # Extract specific time steps
            # t = 4 (labeled frame), t_prev = 3
            t_idx = 4
            prev_idx = 3

            t11_t = band_11[..., t_idx]
            t14_t = band_14[..., t_idx]
            t15_t = band_15[..., t_idx]

            t11_prev = band_11[..., prev_idx]
            t14_prev = band_14[..., prev_idx]
            t15_prev = band_15[..., prev_idx]

        except Exception as e:
            # Fallback for corrupted/missing files (though verification script showed 0 missing)
            # Return zeros to prevent crashing
            print(f"Error loading {record_id}: {e}")
            dummy = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.IN_CHANNELS),
                dtype=np.float32,
            )
            if self.split == "test":
                return {
                    "image": torch.from_numpy(dummy).permute(2, 0, 1),
                    "record_id": record_id,
                }
            else:
                return {
                    "image": torch.from_numpy(dummy).permute(2, 0, 1),
                    "mask": torch.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                }

        # --- Feature Engineering ---
        # 1. Ash Color for time t
        ash_t = get_ash_color(t11_t, t14_t, t15_t)  # (H, W, 3)

        # 2. Ash Color for time t-1
        ash_prev = get_ash_color(t11_prev, t14_prev, t15_prev)  # (H, W, 3)

        # 3. Temporal Difference
        ash_diff = ash_t - ash_prev  # (H, W, 3)

        # 4. Concatenate -> 6 Channels
        # Input shape: (H, W, 6)
        image = np.concatenate([ash_t, ash_diff], axis=-1).astype(np.float32)

        # --- Load Mask (Train/Val only) ---
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(Config.INPUT_ROOT, row["human_pixel_masks"])
            # Mask shape (H, W, 1)
            mask = np.load(mask_path).astype(np.float32)

        # --- Augmentations ---
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Convert to Tensor manually if no transform provided (fallback)
            # Albumentations ToTensorV2 handles HWC -> CHW
            transforms_fallback = ToTensorV2()
            if mask is not None:
                augmented = transforms_fallback(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = transforms_fallback(image=image)
                image = augmented["image"]

        # --- Return ---
        sample = {"image": image, "record_id": record_id}  # (6, H, W)

        if mask is not None:
            # Ensure mask is (1, H, W) if not already
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.shape[0] != 1 and mask.shape[-1] != 1:
                # If shape is strange, assume it's correct from ToTensorV2
                pass

            # If mask came from ToTensorV2, it might be (H, W) or (1, H, W) depending on setup
            # We want (1, H, W) for BCE/Dice
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.shape[-1] == 1:  # (H, W, 1) -> (1, H, W)
                mask = mask.permute(2, 0, 1)

            sample["mask"] = mask

        return sample


def get_transforms(split="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        split (str): 'train' or 'validation'/'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloader(
    split, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle=None
):
    """
    Factory function to create a DataLoader.

    Args:
        split (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = ContrailDataset(
        split=split, transform=get_transforms(split), debug=Config.DEBUG
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(
            split == "train"
        ),  # Drop incomplete batch in training to maintain batch stats
    )

    return loader

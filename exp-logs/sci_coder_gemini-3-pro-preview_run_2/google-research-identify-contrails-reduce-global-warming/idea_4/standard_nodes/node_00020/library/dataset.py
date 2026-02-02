import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import get_ash_colors


def get_transforms(split):
    """
    Returns the Albumentations composition based on the dataset split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Strict Affine transformations: Rotation, Scale, Shift.
                # Avoid elastic/grid distortions which warp linear contrails.
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"mask": "image"},
        )
    else:
        # Validation and Test: No geometric augmentations, just tensor conversion
        return A.Compose(
            [ToTensorV2(transpose_mask=True)], additional_targets={"mask": "image"}
        )


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.

    Loads satellite imagery bands, constructs the 6-channel input (Ash Color + Temporal Diff),
    and applies affine augmentations.
    """

    def __init__(self, split="train", max_samples=None, transform=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            max_samples (int, optional): Limit dataset size for debugging.
            transform (A.Compose, optional): Custom transforms. If None, uses default based on split.
        """
        self.split = split
        self.input_dir = Config.INPUT_DIR

        # Select metadata file based on split
        if split == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Debugging: Limit samples
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()

        # Set transforms
        if transform is None:
            self.transform = get_transforms(split)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # ---------------------------------------------------------
        # 1. Load Satellite Bands
        # ---------------------------------------------------------
        # We need bands 11, 14, 15 for Ash Color Composite
        # Paths are relative in metadata, e.g., "train/ID/band_11.npy"

        try:
            # Load raw data: Shape (H, W, T)
            # T = n_times_before(4) + n_times_after(3) + 1 = 8 frames
            b11 = np.load(os.path.join(self.input_dir, row["band_11"]))
            b14 = np.load(os.path.join(self.input_dir, row["band_14"]))
            b15 = np.load(os.path.join(self.input_dir, row["band_15"]))
        except Exception as e:
            print(f"Error loading bands for record {record_id}: {e}")
            # Return zeros in case of read error to avoid crashing
            dummy_img = torch.zeros(
                (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=torch.float32,
            )
            dummy_mask = torch.zeros(
                (1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
            )
            return dummy_img, dummy_mask, record_id

        # ---------------------------------------------------------
        # 2. Input Engineering (6 Channels)
        # ---------------------------------------------------------
        # Labeled frame is at index 4 (t=4)
        # Previous frame is at index 3 (t=3)
        idx_t4 = 4
        idx_t3 = 3

        # Extract specific time steps
        # Shape becomes (H, W)
        b11_t4 = b11[..., idx_t4]
        b14_t4 = b14[..., idx_t4]
        b15_t4 = b15[..., idx_t4]

        b11_t3 = b11[..., idx_t3]
        b14_t3 = b14[..., idx_t3]
        b15_t3 = b15[..., idx_t3]

        # Compute Ash False Color Composite (H, W, 3)
        # Returns values in [0, 1]
        ash_t4 = get_ash_colors(b11_t4, b14_t4, b15_t4)
        ash_t3 = get_ash_colors(b11_t3, b14_t3, b15_t3)

        # Compute Temporal Difference
        # Channels 4-6: Ash(t=4) - Ash(t=3)
        # Range will be approx [-1, 1]
        diff = ash_t4 - ash_t3

        # Concatenate to create 6-channel input
        # Shape: (H, W, 6)
        img = np.concatenate([ash_t4, diff], axis=-1)

        # Ensure float32
        img = img.astype(np.float32)

        # ---------------------------------------------------------
        # 3. Load Mask
        # ---------------------------------------------------------
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(self.input_dir, row["human_pixel_masks"])
            # Load mask: Shape (H, W, 1)
            mask = np.load(mask_path).astype(np.float32)
        else:
            # Test set: create dummy mask
            h, w = img.shape[:2]
            mask = np.zeros((h, w, 1), dtype=np.float32)

        # ---------------------------------------------------------
        # 4. Augmentation
        # ---------------------------------------------------------
        if self.transform:
            # Albumentations expects HWC
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

            # Albumentations ToTensorV2 converts to (C, H, W)
            # Mask might come out as (H, W) or (1, H, W) depending on setup
            # Ensure mask is (1, H, W)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.shape[0] != 1 and mask_tensor.shape[-1] != 1:
                # If it's somehow (H, W, 1) but tensor, permute
                # But ToTensorV2 usually handles HWC -> CHW
                pass
        else:
            # Manual conversion if no transform provided
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        return img_tensor, mask_tensor, record_id

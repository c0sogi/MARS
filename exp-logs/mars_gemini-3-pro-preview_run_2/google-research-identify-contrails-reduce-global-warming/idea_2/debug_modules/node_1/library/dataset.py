import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VALIDATION_METADATA_PATH,
    TEST_METADATA_PATH,
    IMG_SIZE,
    TIME_CURRENT,
    TIME_PREV,
    TIME_NEXT,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Identification.

    Implements the Symmetric Temporal-Difference strategy:
    - Input: 9 Channels
        - Ch 1-3: Ash Composite at t=Current
        - Ch 4-6: Ash Composite Difference (Current - Prev)
        - Ch 7-9: Ash Composite Difference (Current - Next)
    - Target: Binary Segmentation Mask
    """

    def __init__(self, split="train", transform=None, debug=DEBUG):
        self.split = split
        self.transform = transform

        # Select metadata file based on split
        if split == "train":
            self.meta_path = TRAIN_METADATA_PATH
        elif split == "validation":
            self.meta_path = VALIDATION_METADATA_PATH
        elif split == "test":
            self.meta_path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        # Handle Debug Mode
        if debug:
            self.df = self.df.sample(
                n=min(len(self.df), DEBUG_SAMPLE_SIZE), random_state=42
            ).reset_index(drop=True)

        # Define normalization bounds for Ash Composite (Brightness Temperature in Kelvin)
        # Based on GOES-16 ABI standard practices for Contrail/Ash detection
        self.bounds = {
            "T11": (243, 303),
            "T14": (243, 303),
            "T15": (243, 303),
            "T15_T14": (-4, 2),  # Red Channel Range
            "T14_T11": (-4, 5),  # Green Channel Range
        }

    def normalize(self, data, min_v, max_v):
        """Linearly normalizes data to [0, 1] based on provided bounds."""
        return (data - min_v) / (max_v - min_v)

    def get_ash_composite(self, b11, b14, b15):
        """
        Generates a 3-channel Ash False Color Composite.

        Args:
            b11, b14, b15: 2D arrays of brightness temperatures.

        Returns:
            np.ndarray: (H, W, 3) float32 array in range [0, 1].
        """
        # Red: T15 - T14
        r = self.normalize(
            b15 - b14, self.bounds["T15_T14"][0], self.bounds["T15_T14"][1]
        )

        # Green: T14 - T11
        g = self.normalize(
            b14 - b11, self.bounds["T14_T11"][0], self.bounds["T14_T11"][1]
        )

        # Blue: T14
        b = self.normalize(b14, self.bounds["T14"][0], self.bounds["T14"][1])

        # Clip to ensure valid range [0, 1]
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)

        return np.stack([r, g, b], axis=-1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Satellite Bands
        # Metadata contains relative paths (e.g., "train/123/band_11.npy")
        try:
            path_11 = os.path.join(INPUT_DIR, row["band_11"])
            path_14 = os.path.join(INPUT_DIR, row["band_14"])
            path_15 = os.path.join(INPUT_DIR, row["band_15"])

            # Load raw NPY files (Shape: H x W x T)
            # We assume T=8 (indices 0-7)
            b11_raw = np.load(path_11)
            b14_raw = np.load(path_14)
            b15_raw = np.load(path_15)

        except Exception as e:
            # Fallback for data loading errors
            print(f"Error loading record {record_id}: {e}")
            return torch.zeros((9, IMG_SIZE, IMG_SIZE)), torch.zeros(
                (1, IMG_SIZE, IMG_SIZE)
            )

        # 2. Extract Temporal Slices
        # We need Current (t=4), Previous (t=3), and Next (t=5)
        def get_slice(t):
            return b11_raw[..., t], b14_raw[..., t], b15_raw[..., t]

        b11_c, b14_c, b15_c = get_slice(TIME_CURRENT)
        b11_p, b14_p, b15_p = get_slice(TIME_PREV)
        b11_n, b14_n, b15_n = get_slice(TIME_NEXT)

        # 3. Compute Ash Composites
        ash_curr = self.get_ash_composite(b11_c, b14_c, b15_c)  # (H, W, 3)
        ash_prev = self.get_ash_composite(b11_p, b14_p, b15_p)
        ash_next = self.get_ash_composite(b11_n, b14_n, b15_n)

        # 4. Compute Temporal Differences
        # Captures dynamics: Arrival (Curr - Prev) and Departure (Curr - Next)
        diff_prev = ash_curr - ash_prev
        diff_next = ash_curr - ash_next

        # 5. Construct Input Tensor
        # Stack along channel axis: [Ash_Curr, Diff_Prev, Diff_Next]
        # Shape: (H, W, 9)
        img = np.concatenate([ash_curr, diff_prev, diff_next], axis=-1).astype(
            np.float32
        )

        # 6. Load Mask (if available)
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(INPUT_DIR, row["human_pixel_masks"])
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(np.float32)

        # 7. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            # Default conversion to tensor (HWC -> CHW)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            mask = torch.from_numpy(mask.transpose(2, 0, 1))

        return img, mask


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for the specified split.
    Enforces strict affine transformations for training (no elastic distortion).
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Strict affine transformations: Shift, Scale, Rotate
                # Border mode 0 (Constant) fills new pixels with 0 (Black/No-Change)
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=0,
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation/Test: Only convert to tensor
        return A.Compose([ToTensorV2(transpose_mask=True)])


def get_dataloader(split, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, debug=DEBUG):
    """
    Factory function to create DataLoaders.
    """
    dataset = ContrailDataset(split=split, transform=get_transforms(split), debug=debug)

    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )

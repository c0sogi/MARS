import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(stage: str = "train"):
    """
    Returns the Albumentations transform pipeline for the specified stage.

    Args:
        stage (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if stage == "train":
        return A.Compose(
            [
                # Strict Affine Transformation: Rotation, Scale, Shift
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=0,  # Constant padding
                    value=0,  # Pad with 0
                ),
                # Flips
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No geometric augmentations, just tensor conversion
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.
    Loads satellite bands, constructs 6-channel input (Ash + Temporal Diff),
    and applies augmentations.
    """

    def __init__(self, split="train", max_samples=None, transform=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            max_samples (int, optional): Limit dataset size for debugging.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None, defaults are used based on split.
        """
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split)

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

        # Debugging: Limit samples if requested
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # Pre-compute full paths for efficiency
        # We need bands 11, 13, 14, 15 for features
        self.band_paths = {}
        for b in [11, 13, 14, 15]:
            col_name = f"band_{b}"
            # Join with INPUT_DIR
            self.band_paths[b] = [
                os.path.join(Config.INPUT_DIR, p) for p in self.df[col_name]
            ]

        # Mask paths (only for train/validation)
        self.has_masks = split in ["train", "validation"]
        if self.has_masks:
            self.mask_paths = [
                os.path.join(Config.INPUT_DIR, p) for p in self.df["human_pixel_masks"]
            ]
        else:
            self.mask_paths = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Shape (6, H, W)
            mask (torch.Tensor): Shape (1, H, W) (if available)
            record_id (str): ID of the sample
        """
        # 1. Load required bands
        # Shape of npy files: H x W x T (T=8)
        # We need T=4 (labeled frame) and T=3 (previous frame)

        bands_data = {}
        try:
            for b in [11, 13, 14, 15]:
                # Load full sequence
                # Optimization: We could use mmap_mode='r' if memory is tight,
                # but for 256x256x8 floats, standard load is fine.
                full_seq = np.load(self.band_paths[b][idx])

                # Extract specific timesteps
                bands_data[f"b{b}_t4"] = full_seq[..., Config.LABELED_FRAME_IDX]
                bands_data[f"b{b}_t3"] = full_seq[..., Config.PREV_FRAME_IDX]
        except Exception as e:
            # Fallback for corrupt files (though metadata validation should catch this)
            print(f"Error loading bands for index {idx}: {e}")
            # Return zero tensors
            return torch.zeros(
                (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            ), torch.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # 2. Construct Ash False Color Composite (Channels 1-3)
        # Formulae based on brightness temperatures

        # Red: Band 15 - Band 13
        r = bands_data["b15_t4"] - bands_data["b13_t4"]
        r = (r - Config.ASH_RED_MIN) / (Config.ASH_RED_MAX - Config.ASH_RED_MIN)

        # Green: Band 14 - Band 11
        g = bands_data["b14_t4"] - bands_data["b11_t4"]
        g = (g - Config.ASH_GREEN_MIN) / (Config.ASH_GREEN_MAX - Config.ASH_GREEN_MIN)

        # Blue: Band 13
        b = bands_data["b13_t4"]
        b = (b - Config.ASH_BLUE_MIN) / (Config.ASH_BLUE_MAX - Config.ASH_BLUE_MIN)

        # Clip to [0, 1]
        ash_composite = np.stack([r, g, b], axis=-1)
        ash_composite = np.clip(ash_composite, 0, 1)

        # 3. Construct Raw Temporal Differences (Channels 4-6)
        # t=4 minus t=3 for bands 11, 14, 15
        # We do NOT normalize these, preserving raw dynamic range as per strategy
        diff_11 = bands_data["b11_t4"] - bands_data["b11_t3"]
        diff_14 = bands_data["b14_t4"] - bands_data["b14_t3"]
        diff_15 = bands_data["b15_t4"] - bands_data["b15_t3"]

        temporal_diffs = np.stack([diff_11, diff_14, diff_15], axis=-1)

        # 4. Concatenate to form 6-channel input
        # Shape: (H, W, 6)
        image = np.concatenate([ash_composite, temporal_diffs], axis=-1).astype(
            np.float32
        )

        # 5. Load Mask (if available)
        mask = None
        if self.has_masks:
            try:
                # Shape: H x W x 1
                mask = np.load(self.mask_paths[idx]).astype(np.float32)
            except Exception as e:
                print(f"Error loading mask for index {idx}: {e}")
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # 6. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask is channel-first: (H, W, 1) -> (1, H, W)
                # Albumentations ToTensorV2 converts image to (C, H, W) but mask handling varies.
                # If mask is 2D (H, W), ToTensorV2 makes it (H, W).
                # If mask is 3D (H, W, 1), ToTensorV2 makes it (1, H, W) usually if passed as mask?
                # Actually ToTensorV2 doesn't transpose mask if it's passed as 'mask'.
                # It usually returns a tensor. Let's ensure shape.
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:  # (H, W, 1)
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # If test set, we don't return mask
        if not self.has_masks:
            return image, str(self.df.iloc[idx]["record_id"])

        return image, mask

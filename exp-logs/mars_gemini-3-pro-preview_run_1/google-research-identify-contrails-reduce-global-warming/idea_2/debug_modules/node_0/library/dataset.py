import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.

    Handles:
    1. Loading specific spectral bands (11, 14, 15) from NPY files.
    2. Generating Ash Color Composite with physics-based normalization.
    3. Loading binary ground truth masks (for train/validation).
    4. Applying discrete geometric augmentations (Flip, Rotate90).
    """

    def __init__(self, split="train", transform=None, debug_subset_size=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None and split='train', defaults are applied.
            debug_subset_size (int, optional): If set, limits dataset size for debugging.
        """
        self.split = split
        self.debug_subset_size = debug_subset_size

        # 1. Load Metadata
        metadata_file = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.df = pd.read_csv(metadata_file)

        # Handle debugging subset
        if self.debug_subset_size is not None:
            self.df = self.df.iloc[: self.debug_subset_size]

        # 2. Define Transforms
        # If no transform is provided and we are training, use default geometric augs
        if transform is None and split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ]
            )
        elif transform is None:
            # For validation/test, just convert to tensor
            self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # ----------------------------------------------------------------------
        # 1. Load Image Bands (Ash Composite Construction)
        # ----------------------------------------------------------------------
        # We need bands 11, 14, and 15.
        # Paths in metadata are relative to Config.INPUT_DIR

        try:
            # Load full temporal sequences (H, W, T)
            path_b11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            path_b14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            path_b15 = os.path.join(Config.INPUT_DIR, row["band_15"])

            # Load NPY and extract the labeled frame
            # Config.LABELED_FRAME_IDX is usually 4 (5th frame)
            t_idx = Config.LABELED_FRAME_IDX

            # Use mmap_mode='r' if files are huge, but they are 2MB, so standard load is fine
            b11 = np.load(path_b11)[..., t_idx].astype(np.float32)
            b14 = np.load(path_b14)[..., t_idx].astype(np.float32)
            b15 = np.load(path_b15)[..., t_idx].astype(np.float32)

            # ------------------------------------------------------------------
            # 2. Normalize and Create Ash Composite
            # ------------------------------------------------------------------
            # Red: Optical Depth Proxy (Band 15 - Band 14)
            r = self._normalize(b15 - b14, Config.ASH_RED_MIN, Config.ASH_RED_MAX)

            # Green: Particle Phase Proxy (Band 14 - Band 11)
            g = self._normalize(b14 - b11, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX)

            # Blue: Temperature (Band 14)
            b = self._normalize(b14, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

            # Stack to (H, W, 3)
            image = np.stack([r, g, b], axis=-1)

        except Exception as e:
            # Fallback for corrupt data (should not happen in clean dataset)
            print(f"Error loading record {record_id}: {e}")
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # ----------------------------------------------------------------------
        # 3. Load Mask (if available)
        # ----------------------------------------------------------------------
        mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        if self.split != "test":
            mask_path_rel = row.get("human_pixel_masks")
            if pd.notna(mask_path_rel):
                mask_path = os.path.join(Config.INPUT_DIR, mask_path_rel)
                if os.path.exists(mask_path):
                    # Mask is (H, W, 1) -> squeeze to (H, W)
                    mask = np.load(mask_path).astype(np.float32).squeeze()

        # ----------------------------------------------------------------------
        # 4. Apply Augmentations
        # ----------------------------------------------------------------------
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W) if it came out as (H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # For test set, we might want to return record_id, but standard loaders
        # usually expect (input, target). We return (image, mask) here.
        # The inference loop can access record_ids via dataset.df.
        return image, mask

    def _normalize(self, data, min_val, max_val):
        """
        Linearly normalizes data to [0, 1] based on physical bounds.
        Clips values outside the range.
        """
        data = (data - min_val) / (max_val - min_val)
        return np.clip(data, 0.0, 1.0)

    def get_record_id(self, idx):
        """Helper to get record_id for a specific index (useful for submission)."""
        return str(self.df.iloc[idx]["record_id"])

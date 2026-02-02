import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import INPUT_DIR, ASH_BOUNDS, IMG_SIZE, N_CHANNELS
from library.utils import normalize_range, get_transforms


class ContrailsDataset(Dataset):
    def __init__(self, metadata_path, split="train", transform=None, debug_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'validation', or 'test'.
            transform (A.Compose, optional): Albumentations transform pipeline.
            debug_size (int, optional): If provided, limits the dataset to this many samples.
        """
        self.split = split
        self.transform = transform or get_transforms(data=split)

        # Load metadata
        try:
            self.df = pd.read_csv(metadata_path)
            # Ensure record_id is treated as string
            self.df["record_id"] = self.df["record_id"].astype(str)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        if debug_size is not None:
            self.df = self.df.iloc[:debug_size]

        # Pre-construct file paths relative to INPUT_DIR
        # We only need bands 11, 14, 15 for Ash Composite
        self.records = self.df["record_id"].values
        self.band_11_paths = self.df["band_11"].values
        self.band_14_paths = self.df["band_14"].values
        self.band_15_paths = self.df["band_15"].values

        self.has_masks = "human_pixel_masks" in self.df.columns
        if self.has_masks:
            self.mask_paths = self.df["human_pixel_masks"].values
        else:
            self.mask_paths = None

    def __len__(self):
        return len(self.df)

    def _load_band(self, rel_path):
        """Loads a band file from the input directory."""
        full_path = os.path.join(INPUT_DIR, rel_path)
        # Shape: (H, W, T)
        return np.load(full_path)

    def _compute_ash_composite(self, t11, t14, t15):
        """
        Computes the Ash False Color Composite.

        Args:
            t11, t14, t15 (np.ndarray): Brightness temperature arrays (H, W).

        Returns:
            np.ndarray: Normalized Ash composite (H, W, 3).
        """
        # Red: T15 - T14
        r = normalize_range(
            t15 - t14, ASH_BOUNDS["T15_T14_MIN"], ASH_BOUNDS["T15_T14_MAX"]
        )

        # Green: T14 - T11
        g = normalize_range(
            t14 - t11, ASH_BOUNDS["T14_T11_MIN"], ASH_BOUNDS["T14_T11_MAX"]
        )

        # Blue: T14
        b = normalize_range(t14, ASH_BOUNDS["T14_MIN"], ASH_BOUNDS["T14_MAX"])

        # Stack along last dimension -> (H, W, 3)
        return np.dstack((r, g, b))

    def __getitem__(self, idx):
        record_id = self.records[idx]

        # 1. Load Bands (H, W, T)
        # We need T=3 (previous) and T=4 (current/labeled)
        # Indices are 0-based, so 4th image is index 3, 5th is index 4.
        # Dataset description: "n_times_before=4", so labeled frame is index 4.

        b11_all = self._load_band(self.band_11_paths[idx])
        b14_all = self._load_band(self.band_14_paths[idx])
        b15_all = self._load_band(self.band_15_paths[idx])

        # Extract specific time steps
        # Current frame (t=4)
        t11_curr = b11_all[..., 4]
        t14_curr = b14_all[..., 4]
        t15_curr = b15_all[..., 4]

        # Previous frame (t=3)
        t11_prev = b11_all[..., 3]
        t14_prev = b14_all[..., 3]
        t15_prev = b15_all[..., 3]

        # 2. Compute Features
        # Ash Composite for Current Frame
        ash_curr = self._compute_ash_composite(t11_curr, t14_curr, t15_curr)

        # Ash Composite for Previous Frame
        ash_prev = self._compute_ash_composite(t11_prev, t14_prev, t15_prev)

        # Temporal Difference (Current - Previous)
        # We use the difference of the Ash features to maintain feature space consistency
        diff_ash = ash_curr - ash_prev

        # Concatenate to 6 channels: [Ash_R, Ash_G, Ash_B, Diff_R, Diff_G, Diff_B]
        # Shape: (H, W, 6)
        image = np.concatenate([ash_curr, diff_ash], axis=-1)

        # 3. Load Mask
        mask = None
        if self.has_masks:
            mask_path = os.path.join(INPUT_DIR, self.mask_paths[idx])
            # Load mask: (H, W, 1) -> Squeeze to (H, W)
            mask = np.load(mask_path).squeeze()
        else:
            # Placeholder mask for test set (H, W)
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # 4. Augmentations
        # Albumentations expects 'image' and 'mask'
        # image is (H, W, C), mask is (H, W)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Becomes Tensor (C, H, W) via ToTensorV2
            mask = augmented[
                "mask"
            ]  # Becomes Tensor (H, W) or (1, H, W) depending on config

        # Ensure mask has channel dimension (1, H, W) if it came out as (H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Cast to float
        image = image.float()
        mask = mask.float()

        return {"image": image, "mask": mask, "record_id": record_id}
